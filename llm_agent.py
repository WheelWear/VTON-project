import os
import requests
import openai
import json
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app_agent.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# OpenAI 클라이언트 초기화 (OpenAI API 사용)
client = openai.OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)

# SerpAPI 설정
SERPAPI_KEY = os.environ.get("SERPAPI_KEY")
SERPAPI_URL = "https://serpapi.com/search"

# SerpAPI를 통해 웹 검색 수행 함수
def search_brand_trends(data):
    """
    SerpAPI를 사용해 브랜드의 의류 사이즈 경향에 대한 웹 검색을 수행하고,
    제목, 링크, 요약을 포함한 결과를 반환합니다.
    - Gender: {data.gender}
    - Brand: {data.brand}
    - Cloth type: {data.cloth_type}
    """
    params = {
        "q": f"{data.gender} {data.cloth_type} {data.brand} clothing for size trends",
        "api_key": SERPAPI_KEY,
        "num": 3  # 상위 3개 검색 결과
    }
    response = requests.get(SERPAPI_URL, params=params)
    if response.status_code == 200:
        results = response.json().get("organic_results", [])
        search_data = []
        for result in results:
            title = result.get("title", "No title")
            link = result.get("link", "No link")
            snippet = result.get("snippet", "No snippet")
            search_data.append({
                "title": title,
                "link": link,
                "snippet": snippet
            })
        return search_data
    return [{"title": "No relevant information found.", "link": "", "snippet": ""}]

# OpenAI API를 호출하는 함수
def get_llm_response(prompt: str) -> str:
    """
    OpenAI API를 통해 LLM에게 프롬프트를 보내고 응답을 받습니다.
    응답 형식을 JSON 객체로 설정합니다.
    """
    response = client.chat.completions.create(
        model="chatgpt-4o-latest",  # OpenAI에서 제공하는 모델
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=200,
        temperature=0.7,
        response_format={"type": "json_object"}  # JSON 객체로 응답하도록 설정
    )
    return response.choices[0].message.content

# 사이즈 추천 함수
def recommend_size(data):
    """
    사용자의 신체 치수와 브랜드 정보를 바탕으로 사이즈를 추천하고 설명을 제공합니다.
    웹 검색 결과의 출처(링크)도 함께 반환합니다.
    """
    # SerpAPI로 브랜드 검색
    search_results = search_brand_trends(data)

    # 검색 결과를 프롬프트에 포함할 문자열 생성
    search_summary = "\n".join([
        f"- Title: {result['title']}\n  Link: {result['link']}\n  Snippet: {result['snippet']}"
        for result in search_results
    ])
    logging.info(f"Search results: \n{search_results}")
    # LLM에 보낼 프롬프트 작성
    prompt = f"""
    You are a clothing size recommendation expert. Based on the following user data and web search results, recommend an appropriate size for the {data.brand} brand and provide an explanation. Finally reply in Korean.

    - Gender: {data.gender}
    - Brand: {data.brand}
    - Cloth type: {data.cloth_type}
    - Usual size: {data.cloth_size}
    - Body measurements:
      - Chest circumference: {data.chest_circumference} cm
      - Shoulder width: {data.shoulder_width} cm
      - Arm length: {data.arm_length} cm
      - Waist circumference: {data.waist_circumference} cm

    Web search results for "{data.brand} clothing size trends recent reviews":
    {search_summary}

    Please respond in JSON format with the following fields:
    - "recommend_size": (string) The recommended size (e.g., "M", "L", "36인치")
    - "additional_explanation": (string) Explanation for the recommendation in Korean

    Example response:
    {{"recommend_size": "M", "additional_explanation": "이 브랜드는 보통 작게 나온다는 리뷰가 많아, 사용자의 치수를 고려할 때 M이 적합합니다."}}
    """
    logger.info(f"Prompt: \n{prompt}")
    
    # OpenAI API 호출
    llm_response = get_llm_response(prompt)
    logger.info(f"LLM Response: \n{llm_response}")
    
    # JSON 응답 파싱
    try:
        response_data = json.loads(llm_response)
        recommend_size = response_data["recommend_size"]
        additional_explanation = response_data["additional_explanation"]
    except (json.JSONDecodeError, KeyError):
        recommend_size = "Unknown"
        additional_explanation = "사이즈를 추천하기에 정보가 부족합니다."
    
    # 검색 결과의 링크 추출
    references = [result['link'] for result in search_results if result['link']]

    return {
        "recommend_size": recommend_size,
        "additional_explanation": additional_explanation,
        "references": references,
        "reference_num": len(references)
    }