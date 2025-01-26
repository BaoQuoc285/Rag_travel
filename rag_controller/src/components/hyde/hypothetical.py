import logging
from typing import List, Optional
from langchain.schema import Document
from langchain_core.language_models import BaseLanguageModel
from typing import Any, List

logger = logging.getLogger(__name__)

class HyDe:
    """Hypothetical Document Generation for improved retrieval"""
    
    def __init__(self, llm: BaseLanguageModel):
        self.llm = llm
        self.hypothesis_prompts = {
            'vi': """Nếu câu hỏi KHÔNG liên quan đến du lịch, lịch sử, văn hóa, địa điểm, ẩm thực, giáo dục hoặc thông tin hướng dẫn của TPHCM, hãy trả về "OFF_TOPIC".

Với các trường học ở TPHCM, hãy xem như một phần của địa điểm tham quan và di tích lịch sử.

Nếu liên quan, và từ ngữ không dấu hoặc sai chỉnh tả hãy thêm dấu hoặc sửa chính tả nếu có thể thành tên địa điểm. Sau đó hãy viết một đoạn văn ngắn giới thiệu lịch sử hoặc miêu tả ngắn địa điểm hoặc trả lời câu hỏi, như thể bạn là một hướng dẫn viên du lịch chuyên nghiệp ở TPHCM:

Câu hỏi: {question}

Yêu cầu:
- Nếu nhắc tên riêng mà không nói gì thì mặc định là địa điểm ở TPHCM.
- Chỉ trả lời các chủ đề liên quan TPHCM
- Nếu kêu giới thiệu địa điểm thì lặp tên địa điểm 5 lần rồi mới giới thiệu
- Viết ngắn gọn, súc tích (2-3 câu)
- Không cần trích dẫn nguồn
- Viết như một phần của cẩm nang du lịch"""
        }

    async def generate_hypothetical_answer(self, question: str, lang: str = 'vi') -> str:
        """Generate a hypothetical answer to be used for retrieval"""
        try:
            #replace sai gon with thanh pho ho chi minh
            if "gia định" not in question and "sài gòn" in question:
                question = question.replace("sài gòn", "Thành phố Hồ Chí Minh")
            # Always use Vietnamese prompt
            prompt = self.hypothesis_prompts['vi'].format(question=question)
            response = await self.llm.ainvoke(prompt)
            hypothetical_answer = response.content if hasattr(response, 'content') else str(response)
            if "OFF_TOPIC" in hypothetical_answer:
                logger.info("Query classified as off-topic")
                return None
            logger.info(f"Generated hypothetical answer: {hypothetical_answer}")
            return hypothetical_answer
        except Exception as e:
            logger.error(f"Error generating hypothetical answer: {e}")
            return question  # Fallback to original question if generation fails

    async def summarize_history(self, chat_history: List[str], lang: str) -> str:
        """Summarize recent chat history into a single context sentence."""
        if not chat_history:
            return ""
            
        prompt = f"""Summarize this conversation history in a single sentence that captures the main topic:

        Chat History:
        {chat_history}
        
        Give a concise summary in 'Vietnamese in one sentence'.
        """
        
        try:
            response = await self.llm.ainvoke(prompt)
            return response.content
        except Exception as e:
            logger.error(f"Error summarizing history: {e}")
            return ""

    async def enhanced_retrieval(self, 
                               retriever: Any, 
                               question: str, 
                               lang: str = 'en',
                               k: int = 3,
                               chat_history: List[str] = None,
                               stored_context: str = None
                               ) -> List[Document]:
        """Perform enhanced retrieval using hypothetical answer and chat history"""
        try:
            # Extract city context from history or question
            city_context = "Thành phố Hồ Chí Minh"
            if "sài gòn" in question.lower():
                city_context = "Sài Gòn"
            
            # Handle general topics (weather, climate, etc.)
            general_topics = ["thời tiết", "khí hậu", "mùa", "season", "weather", "climate"]
            if any(topic in question.lower() for topic in general_topics):
                if not stored_context or "sài gòn" not in stored_context.lower():
                    question = f"{city_context} {question}"
                    logger.info(f"Added city context to general topic: {question}")

            # Build enhanced question with all context
            enhanced_question = question
            if stored_context:
                if "sài gòn" in stored_context.lower() or "hồ chí minh" in stored_context.lower():
                    # Already has city context, just combine with question
                    enhanced_question = f"{stored_context} - {question}"
                else:
                    # Add city context if needed
                    enhanced_question = f"{city_context} {stored_context} - {question}"
                    
                logger.info(f"Enhanced question with context: {enhanced_question}")

            # If we have chat history, use it to further enhance the question
            if chat_history:
                history_summary = await self.summarize_history(chat_history, lang)
                if history_summary:
                    logger.info(f"Adding history context: {history_summary}")
                    enhanced_question = f"{enhanced_question} ({history_summary})"

            # Generate hypothetical answer based on enhanced question
            hyde_query = await self.generate_hypothetical_answer(enhanced_question, lang)
            if not hyde_query:
                return []

            # Build final query combining all context
            final_query = f"""Question: {enhanced_question}
            Previous context: {history_summary if chat_history else 'None'}
            Location context: {stored_context if stored_context else 'None'}
            Hypothetical answer: {hyde_query}"""
            
            logger.info(f"Final enhanced query: {final_query}")
            return await retriever.aget_relevant_documents(final_query)

        except Exception as e:
            logger.error(f"Error in enhanced retrieval: {e}")
            return await retriever.aget_relevant_documents(question)
