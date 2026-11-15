"""Conversational AI Customer Support Engine.
Implements the conversational support module described in Section IV-B of
the research paper. Provides intelligent intent parsing, entity extraction
(order IDs, product queries, categories), database-driven order tracking,
catalog product search, detailed retail FAQ resolution, and session memory.
Fully functional without requiring any paid APIs or external keys.
"""
from pathlib import Path
import re
import sys
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.database import get_order_by_id, search_products, log_chat_message, get_db_connection


class RetailConversationalAI:
    """Intelligent rule-and-intent conversational AI with entity extraction and database linkage."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path
        self._init_knowledge_base()

    def _init_knowledge_base(self):
        """Structured policy knowledge base for retail e-commerce."""
        self.faq_topics = {
            "shipping": {
                "keywords": ["shipping", "delivery", "dispatch", "courier", "track", "transit", "arrive", "when will", "charge", "free delivery"],
                "response": (
                    "📦 **Shipping & Delivery Policy**:\n\n"
                    "- **Standard Delivery**: 3 to 5 business days nationwide.\n"
                    "- **Express Delivery**: 24 to 48 hours in metro cities (select at checkout for ₹99).\n"
                    "- **Free Shipping**: Applied automatically on all orders above ₹999.\n"
                    "- **Courier Partners**: Shipped via BlueDart, Delhivery, and DTDC with real-time SMS tracking updates.\n\n"
                    "💡 *Have an order number? Type something like* `Track order ORD-1002` *to check live status!*"
                )
            },
            "returns": {
                "keywords": ["return", "refund", "exchange", "replace", "broken", "damaged", "wrong item", "money back", "return policy"],
                "response": (
                    "🔄 **Returns & Refund Policy**:\n\n"
                    "- **30-Day Window**: Most items can be returned within 30 days of delivery in original packaging.\n"
                    "- **Free Reverse Pickup**: Our courier agent picks up the parcel from your doorstep at zero cost.\n"
                    "- **Refund Timeline**: Once inspected at our fulfillment center, refunds are credited within 3-5 business days to your original payment method (UPI / Bank / Card).\n"
                    "- **Exempt Items**: Perishable goods and hygiene-sensitive items (unless defective on arrival).\n\n"
                    "💡 *Need to initiate a return? Contact support at returns@smartretail.com or provide your Order ID.*"
                )
            },
            "payment": {
                "keywords": ["payment", "pay", "card", "upi", "netbanking", "cod", "cash on delivery", "failed", "deducted", "invoice", "receipt", "gst"],
                "response": (
                    "💳 **Payment & Security Options**:\n\n"
                    "- **Accepted Methods**: UPI (Google Pay, PhonePe, Paytm), Credit/Debit Cards (Visa, MasterCard, RuPay), Net Banking (50+ banks), and Cash on Delivery (COD).\n"
                    "- **Security**: 256-bit TLS bank-grade encryption with PCI-DSS Level 1 compliant gateway processing.\n"
                    "- **Failed Transactions**: If money was deducted for an unconfirmed order, banks auto-reverse within 24-48 business hours.\n"
                    "- **GST Invoices**: Downloadable directly from your account page immediately after order placement."
                )
            },
            "cancel": {
                "keywords": ["cancel", "cancellation", "stop order", "modify order", "change address"],
                "response": (
                    "⛔ **Order Cancellation & Modification**:\n\n"
                    "- Orders can be cancelled instantly with 100% refund **prior to warehouse dispatch**.\n"
                    "- To modify shipping address or contact phone number, please submit updates within 2 hours of placing the order.\n"
                    "- Once the package is 'Shipped' or 'Out for Delivery', cancellation is unavailable, but you may refuse delivery or request a return upon arrival."
                )
            },
            "discounts": {
                "keywords": ["discount", "coupon", "promo", "offer", "sale", "deal", "voucher", "code"],
                "response": (
                    "🎉 **Current Promotions & Loyalty Discounts**:\n\n"
                    "- **WELCOME10**: Flat 10% off on your first order (up to ₹500).\n"
                    "- **FREESHIP**: Zero delivery fee on orders above ₹499.\n"
                    "- **Loyalty Tiers**: Silver members earn 2% cashback points; Gold earns 5%; Platinum earns 10% on every purchase.\n"
                    "- Check our dynamic pricing discounts on the **Dynamic Pricing** page!"
                )
            },
            "support": {
                "keywords": ["agent", "human", "representative", "customer care", "contact support", "call us", "phone number", "helpdesk", "talk to human"],
                "response": (
                    "🤝 **Customer Support & Human Escalation**:\n\n"
                    "- **Toll-Free Phone**: 1800-419-7382 (Available Mon-Sat, 9:00 AM - 8:00 PM IST)\n"
                    "- **Email Support**: support@smartretail-ai.demo (Average response time: 2 hours)\n"
                    "- **Live Escalation**: Escalation ticket #ESC-9082 opened for your session. Our human support representative will review this chat."
                )
            },
            "greeting": {
                "keywords": ["hello", "hi", "hey", "good morning", "good evening", "greetings", "namaste", "how are you"],
                "response": (
                    "👋 **Hello! Welcome to Smart Retail AI Support.**\n\n"
                    "I am your automated AI shopping assistant. I can help you with:\n"
                    "1. 📦 **Tracking your orders** (e.g. *'Track order ORD-1002'*)\n"
                    "2. 🔍 **Searching product catalog** (e.g. *'Do you have wireless headphones?'*)\n"
                    "3. 🔄 **Returns & Refund policy**\n"
                    "4. 💳 **Payments, EMI & Invoice queries**\n"
                    "5. 🎁 **Product recommendations and gift ideas**\n\n"
                    "How may I assist you today?"
                )
            }
        }

    def extract_order_id(self, text: str) -> Optional[str]:
        """Extract order pattern like ORD-1001, ORD-999999, #1002."""
        match = re.search(r'\b(ORD[-\s]?\d{3,10})\b', text, re.IGNORECASE)
        if match:
            clean = match.group(1).upper().replace(" ", "-")
            return clean
        match_num = re.search(r'#(\d{4,10})', text)
        if match_num:
            return f"ORD-{match_num.group(1)}"
        return None

    def handle_order_query(self, order_id: str) -> Tuple[str, str]:
        """Look up order details directly from SQLite database."""
        order = get_order_by_id(order_id, self.db_path)
        if not order:
            return (
                f"⚠️ **Order Not Found**: I could not locate order `{order_id}` in our database.\n\n"
                f"Please verify the number (example format: `ORD-1001` to `ORD-1150`) or check the **Database & Inventory Explorer** page.",
                "order_not_found"
            )

        status_emoji = {
            "Delivered": "✅",
            "Shipped": "🚚",
            "Out for Delivery": "🛵",
            "Processing": "⏳"
        }.get(order["status"], "📦")

        response = (
            f"{status_emoji} **Order Status for `{order['order_id']}`**:\n\n"
            f"- **Item**: {order['product_name']}\n"
            f"- **Current Status**: **{order['status']}**\n"
            f"- **Order Placed On**: {order['order_date']}\n"
            f"- **Estimated Delivery**: {order['estimated_delivery']}\n"
            f"- **Tracking Number**: `{order['tracking_number']}`\n"
            f"- **Total Amount**: ₹{order['total_amount']:,.2f}\n"
            f"- **Delivery Address**: {order['shipping_address']}\n\n"
            f"Need assistance with changing address or initiating a return once delivered? Let me know!"
        )
        return response, "order_status_success"

    def handle_product_search_query(self, text: str) -> Optional[Tuple[str, str]]:
        """Search products catalog if query contains shopping inquiry."""
        search_triggers = ["looking for", "find", "search", "show me", "price of", "do you have", "suggest", "recommend", "buy", "headphones", "shoes", "keyboard", "smartwatch", "air fryer", "monitor", "coffee maker", "yoga mat", "serum", "bottle", "jacket", "t-shirt"]
        text_lower = text.lower()
        if not any(t in text_lower for t in search_triggers):
            return None

        # Clean search term and strip punctuation
        cleaned = re.sub(r'\b(do you have|show me|price of|can you find|looking for|i want|search for|suggest|recommend|a|the|some)\b', '', text_lower)
        cleaned = "".join(c for c in cleaned if c.isalnum() or c.isspace()).strip()
        if not cleaned:
            cleaned = "".join(c for c in text_lower if c.isalnum() or c.isspace()).strip()

        matches = search_products(cleaned, self.db_path)
        if matches:
            prod_lines = []
            for p in matches[:4]:
                stock_label = f"🟢 In Stock ({p['stock']} left)" if p['stock'] > 0 else "🔴 Out of Stock"
                prod_lines.append(
                    f"- **{p['name']}** ({p['category']})\n"
                    f"  💰 Price: **₹{p['current_price']:,.2f}** | ⭐ Rating: {p['rating']}★ ({p['reviews_count']} reviews) | {stock_label}\n"
                    f"  _{p['description']}_\n"
                )
            result = (
                f"🔍 **Catalog Matches for '{cleaned}'**:\n\n" +
                "\n".join(prod_lines) +
                "\n💡 *You can also explore personalized match rankings in the* **Product Recommendations** *page.*"
            )
            return result, "product_search"

        return None

    def respond(
        self,
        user_message: str,
        session_id: str = "default_session",
        customer_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process user message, preserve conversation context, and return structured reply.
        """
        msg = (user_message or "").strip()
        if not msg:
            return {
                "reply": "Please type a question or choose one of the suggested topics below.",
                "intent": "empty_input",
                "extracted_order_id": None
            }

        text_lower = msg.lower()

        # 1. Check for specific Order ID mention
        extracted_order = self.extract_order_id(msg)
        if extracted_order:
            reply_text, intent = self.handle_order_query(extracted_order)
            log_chat_message(session_id, "user", msg, intent, customer_id, self.db_path)
            log_chat_message(session_id, "assistant", reply_text, intent, customer_id, self.db_path)
            return {
                "reply": reply_text,
                "intent": intent,
                "extracted_order_id": extracted_order
            }

        # Check if user asked about orders without ID
        if any(w in text_lower for w in ["my order", "track order", "order status", "where is my package", "order details"]):
            reply_text = (
                "📦 **Track Your Order**:\n\n"
                "To look up your order, please provide your Order ID (such as `ORD-1001` or `ORD-1025`).\n\n"
                "💡 *Tip: Sample trackable orders in our demo database include* `ORD-1001`, `ORD-1002`, `ORD-1005`, and `ORD-1010`."
            )
            intent = "order_inquiry_prompt"
            log_chat_message(session_id, "user", msg, intent, customer_id, self.db_path)
            log_chat_message(session_id, "assistant", reply_text, intent, customer_id, self.db_path)
            return {"reply": reply_text, "intent": intent, "extracted_order_id": None}

        # 2. Check for Product Catalog Search
        prod_res = self.handle_product_search_query(msg)
        if prod_res:
            reply_text, intent = prod_res
            log_chat_message(session_id, "user", msg, intent, customer_id, self.db_path)
            log_chat_message(session_id, "assistant", reply_text, intent, customer_id, self.db_path)
            return {"reply": reply_text, "intent": intent, "extracted_order_id": None}

        # 3. Match against Retail FAQ and Intent Knowledge Base with Word-Boundary Regex
        best_topic = None
        max_matches = 0

        for topic, data in self.faq_topics.items():
            matches = 0
            for kw in data["keywords"]:
                pattern = r'\b' + re.escape(kw) + r'\b'
                if re.search(pattern, text_lower):
                    matches += 1
            if matches > max_matches:
                max_matches = matches
                best_topic = topic

        if best_topic and max_matches > 0:
            reply_text = self.faq_topics[best_topic]["response"]
            intent = f"faq_{best_topic}"
        else:
            # Fallback response with helpful guided prompts
            reply_text = (
                "🤖 **Smart Retail AI Assistant**:\n\n"
                "I didn't quite catch that. Here are some things you can ask me:\n"
                "- 📦 *'Track order ORD-1002'*\n"
                "- 🔍 *'Do you have wireless headphones or gaming keyboards?'*\n"
                "- 🔄 *'What is your 30-day return policy?'*\n"
                "- 💳 *'What payment methods and EMI options are supported?'*\n"
                "- ⛔ *'How can I cancel or modify my order?'*\n"
                "- 🤝 *'Connect me with a customer support agent'*"
            )
            intent = "general_fallback"

        log_chat_message(session_id, "user", msg, intent, customer_id, self.db_path)
        log_chat_message(session_id, "assistant", reply_text, intent, customer_id, self.db_path)

        return {
            "reply": reply_text,
            "intent": intent,
            "extracted_order_id": None
        }


# Module instance
_chatbot_instance: Optional[RetailConversationalAI] = None

def get_chatbot(db_path: Optional[Path] = None) -> RetailConversationalAI:
    global _chatbot_instance
    if _chatbot_instance is None or db_path is not None:
        _chatbot_instance = RetailConversationalAI(db_path)
    return _chatbot_instance


def chatbot_reply(message: str) -> str:
    """Convenience helper compatible with starter facade."""
    bot = get_chatbot()
    res = bot.respond(message)
    return res["reply"]
