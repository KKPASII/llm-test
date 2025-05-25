import pandas as pd
import re
import kss

# --- Configuration ---
CHUNK_SIZE = 20  # 몇 문장 단위로 묶을지
INPUT_CSV_PATH = "./aladin_bestseller.csv"  # 입력 CSV 파일 경로

# --- Choose your tokenizer ---
TOKENIZER_TYPE = "morpheme"  # Options: "word" or "morpheme"

# --- Initialize KoNLPy tokenizer (if chosen) ---
if TOKENIZER_TYPE == "morpheme":
    try:
        from konlpy.tag import Okt
        okt = Okt()
        print("Using Okt morpheme tokenizer.")
    except ImportError:
        print("KoNLPy or Okt not found. Please install it: pip install konlpy")
        print("Falling back to word-based tokenization.")
        TOKENIZER_TYPE = "word"
    except Exception as e:
        print(f"Error initializing Okt tokenizer: {e}")
        print("Falling back to word-based tokenization.")
        TOKENIZER_TYPE = "word"
if TOKENIZER_TYPE == "word":
    print("Using word-based (whitespace) tokenizer.")

# --- Global list to store chunk data (as in your original structure) ---
all_chunk_data_with_metadata = []
max_token_count = 0
min_token_count = 9999
max_sentence_count = 0
min_sentence_count= 9999

# --- Functions ---
try:
    book_csv = pd.read_csv("./aladin_bestseller.csv", dtype=str) ### 파일 경로 확인
except FileNotFoundError:
    print("Error: aladin_bestseller.csv 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
    exit()
except Exception as e:
    print(f"CSV 파일을 읽는 중 오류 발생: {e}")
    exit()


# chunking 함수 - 문장 'CHUNK_SIZE'개씩 chunking
def chunk_sentences(sentence_list, chunk_size):
    all_chunks = []
    
    if not sentence_list: # 빈 리스트
        return all_chunks

    for i in range(0, len(sentence_list), chunk_size):
        selected_sentences_list = sentence_list[i:i + chunk_size] # 리스트 슬라이싱: 인덱스 범위를 넘어가도 오류 없이 가능한 끝까지 가져옴
        sentence_chunk = " ".join(selected_sentences_list)
        all_chunks.append(sentence_chunk)
    
    # 청크가 만들어지지 않은 경우 (e.g. 길이가 1인데 chunk_sizer가 2)
    if not all_chunks:
        all_chunks.append(" ".join(sentence_list)) # 하나의 청크로 만들기
    return all_chunks


# "[]" 형식 문자열 -> list
def string_to_list(s):
    if s == "" or s.strip() == "[]": # 빈 문자열, "[]"
        return []
    
    str_list = re.findall(r"'(.*?)'", s) # ''(작은 따옴표) 안에 있는 모든 문자열이 담긴 리스트 반환
    return str_list


# "{}" 형식 문자열 -> set
def string_to_set(s):
    if s == "" or s.strip() == "{}": # 빈 문자열, "{}"
        return set()
    
    str_list = re.findall(r"'(.*?)'", s)  # ''(작은 따옴표) 안에 있는 모든 문자열이 담긴 리스트 반환
    str_set = set(str_list) # set으로 변환
    return str_set


def print_all_chunk_data_with_metadata():
    global all_chunk_data_with_metadata
    # 만들어진 chunk 확인
    if all_chunk_data_with_metadata:
        for i, data_item in enumerate(all_chunk_data_with_metadata):
            print(f"\nProcessed Chunk {i+1}:")
            #print(f"  ISBN: {data_item.get('isbn', 'N/A')}")
            #print(f"  Title: {data_item.get('title', 'N/A')}")
            #print(f"  Author: {data_item.get('author', 'N/A')}")
            #print(f"  Keywords: {data_item.get('keywords', [])}")
            #print(f"  Genres: {data_item.get('genres', [])}")
            #print(f"  Image URL: {data_item.get('image_url', 'N/A')}")
            print(f"  Chunk Index: {data_item.get('chunk_index', 'N/A')}")
            #print(f"  Chunk Introduction: {data_item.get('chunk_introduction', 'N/A')}")
            print(f"  Token Count: {data_item.get('token_count', 'N/A')}") # --- ADDED: Token count 출력 ---
    else:
        print("생성된 Chunk 데이터가 없습니다.")


def chunk_file_by_line():
    global all_chunk_data_with_metadata, max_token_count, min_token_count
    all_chunk_data_with_metadata = []
    
    for index, row in book_csv.iterrows():
    # data 컬럼 파싱 - row.get('컬럼', '기본값')
        try:
            title = str(row.get('title', ''))
            author = str(row.get('author', ''))
            keyword_list = str(row.get('keyword', '[]'))
            genre_set = str(row.get('genre', '{}'))
            introduction_str = str(row.get('introduction', ''))
        except KeyError as e:
            print(f"컬럼({e})가 없습니다.")
            continue
        except Exception as e:
            print(f"Error: {e}")
            continue

        # 1-1. keyword 파싱
        keywords = string_to_list(keyword_list) # "[]" -> []

        # 1-2. genre 파싱
        genres = string_to_set(genre_set) # "{}" -> {}
        genres_list = list(genres if genres else []) # set을 list로 변환
        
        # --- 메타데이터 토큰 수 계산 준비 ---
        title_tokens = get_token_count(title)
        author_tokens = get_token_count(author)
        intro_token_count = get_token_count(introduction_str)
        
        # keywords는 리스트이므로, 각 키워드를 공백으로 연결한 후 토큰 수 계산
        keywords_text = " ".join(keywords)
        keywords_tokens = get_token_count(keywords_text)
        
        # genres_list는 리스트이므로, 각 장르를 공백으로 연결한 후 토큰 수 계산
        genres_text = " ".join(genres_list)
        genres_tokens = get_token_count(genres_text)
        
        current_metadata_token_count = title_tokens + author_tokens + keywords_tokens + genres_tokens

        # 2. 각 chunk에 메타데이터 연결하여 최종 데이터 생성
        # 현재 도서의 메타데이터
        meta_data = {
            "title": title,
            "author": author,
            "keywords": keywords,
            "genres": genres_list, # set -> list
        }
        
        total_token_count_for_chunk = intro_token_count + current_metadata_token_count
        max_token_count = max(max_token_count, total_token_count_for_chunk)
        min_token_count = min(min_token_count, total_token_count_for_chunk)
        chunk_data = {
            "chunk_index": index + 1,
            "chunk_introduction": introduction_str,
            "token_count": total_token_count_for_chunk,
            **meta_data # 책 공통 메타데이터
        }
        all_chunk_data_with_metadata.append(chunk_data)
            
    
def chunk_file():
    global all_chunk_data_with_metadata, max_token_count, min_token_count, max_sentence_count, min_sentence_count
    all_chunk_data_with_metadata = []
    max_token_count = 0

    for index, row in book_csv.iterrows():
    # data 컬럼 파싱 - row.get('컬럼', '기본값')
        try:
            title = str(row.get('title', ''))
            author = str(row.get('author', ''))
            keyword_list = str(row.get('keyword', '[]'))
            isbn = str(row.get('isbn', '')) # 없애기
            genre_set = str(row.get('genre', '{}'))
            introduction_str = str(row.get('introduction', ''))
            image_url = str(row.get('image', '')) # 고민
        except KeyError as e:
            print(f"컬럼({e})가 없습니다.")
            continue
        except Exception as e:
            print(f"Error: {e}")
            continue

        # 1-1. keyword 파싱
        keywords = string_to_list(keyword_list) # "[]" -> []

        # 1-2. genre 파싱
        genres = string_to_set(genre_set) # "{}" -> {}
        genres_list = list(genres if genres else []) # set을 list로 변환

        # 2. Introduction 텍스트 chunking
        introduction_chunks = [] # 'CHUNK_SIZE'개의 문장을 묶은 chunk들이 담을 리스트

        if introduction_str != "": # introduction 내용이 있으면, 없으면 그대로 빈 리스트
            try:
                # 문장 분리
                sentence_list = kss.split_sentences(introduction_str) # 문장 리스트 - 한 문장이 하나의 요소
                max_sentence_count = max(max_sentence_count, len(sentence_list))
                min_sentence_count = min(min_sentence_count, len(sentence_list))
            except Exception as e:
                print(f"문장 분리 실패: {e}")
                sentence_list = [introduction_str] # 분리 실패 시 하나의 문장으로

            # 분리된 문장이 있으면 chunking
            if sentence_list != []:
                introduction_chunks = chunk_sentences(sentence_list, CHUNK_SIZE) # [] or chunk 리스트 반환

            # 만약, chunk가 만들어지지 않은 경우
            if not introduction_chunks:
                introduction_chunks = [introduction_str] # 하나의 chunk로 만들기
        else:
            introduction_chunks = [""]
        
        # --- 메타데이터 토큰 수 계산 준비 ---
        title_tokens = get_token_count(title)
        author_tokens = get_token_count(author)
        
        # keywords는 리스트이므로, 각 키워드를 공백으로 연결한 후 토큰 수 계산
        keywords_text = " ".join(keywords)
        keywords_tokens = get_token_count(keywords_text)
        
        # genres_list는 리스트이므로, 각 장르를 공백으로 연결한 후 토큰 수 계산
        genres_text = " ".join(genres_list)
        genres_tokens = get_token_count(genres_text)

        current_metadata_token_count = title_tokens + author_tokens + keywords_tokens + genres_tokens

        # 3. 각 chunk에 메타데이터 연결하여 최종 데이터 생성
        meta_data = {
            "title": title,
            "author": author,
            "keywords": keywords,
            "isbn": isbn, # 없애기
            "genres": genres_list, # set -> list
        }

        for i, chunk_introduction in enumerate(introduction_chunks):
            intro_token_count = get_token_count(chunk_introduction)
            total_token_count_for_chunk = intro_token_count + current_metadata_token_count
            max_token_count = max(max_token_count, total_token_count_for_chunk)
            min_token_count = min(min_token_count, total_token_count_for_chunk)
            chunk_data = {
                "chunk_index": i + 1,
                "chunk_introduction": chunk_introduction,
                "token_count": total_token_count_for_chunk,
                **meta_data # 책 공통 메타데이터
            }
            all_chunk_data_with_metadata.append(chunk_data)
    

def get_token_count(text):
    if not text or not text.strip(): # 텍스트가 비어있거나 공백만 있는 경우
        return 0
    if TOKENIZER_TYPE == "morpheme":
        try:
            return len(okt.morphs(text)) # Okt 형태소 분석기 사용
        except Exception as e:
            print(f"Error during morpheme tokenization: {e}. Text: {text[:50]}...")
            return 0 # 오류 발생 시 0 반환
    elif TOKENIZER_TYPE == "word":
        return len(text.split()) # 공백 기준 단어 분리
    return 0 # 기본값


# --- Main execution example ---
if __name__ == "__main__":
    chunk_file_by_line()
    print_all_chunk_data_with_metadata()
    print(f"chunk_size = {CHUNK_SIZE}")
    print(f"min_token_count = {min_token_count}")
    print(f"max_token_count = {max_token_count}")
    #print(f"min_sentence_count = {min_sentence_count}")
    #print(f"max_sentence_count = {max_sentence_count}")