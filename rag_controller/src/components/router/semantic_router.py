import logging
from enum import Enum
from typing import Dict, List, Tuple
from langchain_core.language_models import BaseLanguageModel

logger = logging.getLogger(__name__)

class QueryType(Enum):
    GREETING = "greeting"
    FOLLOW_UP = "follow_up"
    SAIGON = "saigon"
    OUT_OF_SCOPE = "out_of_scope"
    NEEDS_CLARIFICATION = "needs_clarification"  # Add new type

class SemanticRouter:
    def __init__(self, llm: BaseLanguageModel):
        self.llm = llm
        self.route_patterns = {
            'vi': {
                'greeting': [
                    "xin chào", "chào", "hi", "hello", "chào bạn",
                    "bạn là ai", "bạn tên gì", "who are you",
                    "giới thiệu", "introduce"
                ],
                'follow_up': [
                    "còn gì nữa", "kể thêm", "tell me more",
                    "như vậy", "thế còn", "và", "những gì khác",
                    "có thể", "more", "continue", "tiếp tục",
                    "những địa điểm khác"
                ],
                'location_indicators': [
                    "ở", "tại", "nằm ở", "located", "where", "đâu",
                    "địa chỉ", "address", "đường", "quận", "phường"
                ],
                'saigon_terms': [
                    "sài gòn", "hồ chí minh", "hcm", "tphcm",
                    "quận 1", "quận 2", "quận 3", "quận 4",
                    "quận 5", "quận 6", "quận 7", "quận 8",
                    "quận 9", "quận 10", "quận 11", "quận 12",
                    "thủ đức", "bình thạnh", "phú nhuận",
                    "tân bình", "bến thành",
                    # Add educational institutions
                    "trường", "đại học", "học viện", "marie curie",
                    "lê hồng phong", "trần đại nghĩa", "nguyễn thượng hiền",
                    "bách khoa", "rmit", "hutech", "văn lang",
                    # Add historical and revolutionary sites
                    "căn cứ", "rừng sác", "địa đạo", "củ chi",
                    "bến dược", "bến đình", "khu di tích",
                    "chiến khu", "căn cứ kháng chiến",
                    "đồng khởi", "bến nhà rồng", "dinh độc lập",
                    "hội trường thống nhất", "bảo tàng chứng tích chiến tranh",
                    # Add metro-related terms
                    "metro", "ga metro", "tàu điện ngầm", "tuyến metro",
                    "ga bến thành", "metro bến thành", "ga suối tiên",
                    "nhà ga", "tuyến đường sắt", "đường sắt đô thị",
                ],
                'transport_price': [
                    "giá", "tiền", "phí", "bao nhiêu", "nhiêu",
                    "mất bao nhiêu", "tốn", "chi phí",
                    "đi grab", "xe ôm", "taxi", "xe bus",
                    "xe buýt", "đi xe", "phương tiện",
                    "price", "cost", "fare", "fee",
                    "how much", "transport", "ride"
                ],
                'out_of_scope_terms': [
                    "giá vé máy bay", "vé tàu", "book vé",
                    "đặt phòng", "khách sạn", "nhà nghỉ",
                    "airbnb", "booking", "grab", "be",
                    "gojek", "uber", "shopee", "lazada",
                    "tôi là ai", "bạn là ai"
                ],
                'follow_up_indicators': [
                    "còn gì nữa", "kể thêm", "tell me more",
                    "như vậy", "thế còn", "và", "những gì khác",
                    "có thể", "more", "continue", "tiếp tục",
                    "những địa điểm khác", "các địa điểm",
                    "những nơi", "các nơi", "những chỗ", "các chỗ",
                    "nổi tiếng", "phổ biến", "nổi bật", "nổi trội",
                    "đặc biệt", "đặc sắc", "thú vị"
                ],
            }
        }
        
        # Add topic filters
        self.topic_filters = {
            'off_topic': [
                "ai đẹp", "ai xinh", "đẹp trai", "xinh gái",
                "đẹp nhất", "xinh nhất", "giàu nhất", "nghèo nhất",
                "giỏi nhất", "ai là người", "who is the most",
                "best person", "most beautiful","đẹp gái"
            ],
            'tourism_terms': [
                "du lịch", "tham quan", "địa điểm", "điểm đến",
                "di tích", "lịch sử", "văn hóa", "ẩm thực",
                "phương tiện", "đi lại", "khách sạn", "nhà hàng",
                "chợ", "công viên", "bảo tàng", "tourism",
                "travel", "visit", "attraction", "destination",
                # Add education-related terms
                "trường", "học", "giáo dục", "education",
                # Add historical terms
                "khu di tích", "căn cứ", "địa đạo", "chiến khu",
                "di tích lịch sử", "bảo tàng", "memorial",
                "historical site", "monument"
            ]
        }

    async def classify_query(self, query: str, lang: str = 'vi') -> Tuple[QueryType, float]:
        """Classify the query into one of the defined types"""
        query = query.lower()
        patterns = self.route_patterns['vi']
        
        # If query is about Saigon general topics like weather, climate, etc.
        general_topics = ["thời tiết", "khí hậu", "mùa", "season", "weather", "climate"]
        if any(topic in query for topic in general_topics):
            # Check if there's a location context
            if any(term in query for term in patterns['saigon_terms']):
                return QueryType.SAIGON, 0.9
            return QueryType.FOLLOW_UP, 0.9  # Mark as follow-up to use previous context

        words = [w for w in query.split() if len(w) > 1]

        # Add length check
        if len(words) <= 2:
            # If contains Saigon location terms, mark as NEEDS_CLARIFICATION
            if any(term in query for term in patterns['saigon_terms']):
                return QueryType.NEEDS_CLARIFICATION, 1.0

        # First check if query contains off-topic terms
        if any(term in query for term in self.topic_filters['off_topic']):
            return QueryType.OUT_OF_SCOPE, 1.0
            
        # Check if query is tourism-related
        has_tourism = any(term in query for term in self.topic_filters['tourism_terms'])
        
        # If tourism-related and contains Saigon terms, classify as SAIGON
        if has_tourism and any(term in query for term in patterns['saigon_terms']):
            return QueryType.SAIGON, 0.9

        # Check for real-time pricing/booking queries first
        if any(term in query for term in patterns['transport_price']):
            if any(term in query for term in patterns['out_of_scope_terms']):
                return QueryType.OUT_OF_SCOPE, 0.9

        # Check for greetings first
        if any(term in query for term in patterns['greeting']):
            return QueryType.GREETING, 1.0

        # Check if it's a follow-up query
        is_follow_up = False
        # Check for follow-up indicators
        if any(term in query for term in patterns['follow_up_indicators']):
            is_follow_up = True
        # Check if query matches general pattern for requesting more info
        if any(word in ["các", "những", "tất cả", "list", "liệt kê"] for word in words):
            if any(word in ["địa điểm", "nơi", "chỗ", "điểm", "khu"] for word in words):
                is_follow_up = True
        
        if is_follow_up:
            return QueryType.FOLLOW_UP, 0.9

        # Check if query contains location indicators
        has_location = any(term in query for term in patterns['location_indicators'])
        
        # If it's a location query, check if it's about Saigon
        if has_location:
            if any(term in query for term in patterns['saigon_terms']):
                return QueryType.SAIGON, 0.9
            else:
                return QueryType.OUT_OF_SCOPE, 0.7

        # Default to Saigon if contains any Saigon terms
        if any(term in query for term in patterns['saigon_terms']):
            return QueryType.SAIGON, 0.8

        # Use LLM for more complex classification
        prompt = f"""Classify this question into one of these categories:
        1. GREETING - if it's a greeting or asking about who you are
        2. FOLLOW_UP - if it's asking for more information about previous topic
        3. SAIGON - if it's asking about locations/places in Ho Chi Minh City
        4. OUT_OF_SCOPE - if it's asking about places outside of Ho Chi Minh City

        Question: {query}

        Answer with just the category name.
        """
        
        try:
            response = await self.llm.ainvoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Map LLM response to QueryType
            response_lower = response_text.lower()
            if "greeting" in response_lower:
                return QueryType.GREETING, 0.7
            elif "follow" in response_lower:
                return QueryType.FOLLOW_UP, 0.7
            elif "saigon" in response_lower:
                return QueryType.SAIGON, 0.7
            else:
                return QueryType.OUT_OF_SCOPE, 0.7
                
        except Exception as e:
            logger.error(f"Error in LLM classification: {e}")
            return QueryType.SAIGON, 0.5  # Default to Saigon with low confidence
