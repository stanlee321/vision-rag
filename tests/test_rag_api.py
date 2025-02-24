import pytest
import requests
import os
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "http://localhost:8003/v1"
API_TOKEN = os.getenv("API_TOKEN", "1234")

# Add headers with Bearer token
HEADERS = {
    "Authorization": f"Bearer {API_TOKEN}"
}

@pytest.fixture
def test_pdf_path():
    return "../data/2502.06472v1.pdf"

def test_upload_document_smart(test_pdf_path):
    # Test smart loader upload
    files = {'file': open(test_pdf_path, 'rb')}
    response = requests.post(
        f"{BASE_URL}/rag/upload",
        params={'collection_name': 'test_collection_smart', 'loader': 'smart'},
        files=files,
        headers=HEADERS
    )
    assert response.status_code == 200
    assert response.json().get('status') == 'success'

def test_upload_document_low(test_pdf_path):
    # Test low loader upload
    files = {'file': open(test_pdf_path, 'rb')}
    response = requests.post(
        f"{BASE_URL}/rag/upload",
        params={'collection_name': 'test_collection_low', 'loader': 'low'},
        files=files,
        headers=HEADERS
    )
    assert response.status_code == 200
    assert response.json().get('status') == 'success'

def test_query_smart_collection():
    # Test querying smart collection
    params = {
        'q': 'What is the document about?',
        'collection_name': 'test_collection_smart',
        'response_mode': 'compact'
    }
    response = requests.get(
        f"{BASE_URL}/rag/query", 
        params=params,
        headers=HEADERS
    )
    assert response.status_code == 200
    assert 'answer' in response.json()

def test_query_low_collection():
    # Test querying low collection
    params = {
        'q': 'What is the document about?',
        'collection_name': 'test_collection_low',
        'response_mode': 'tree_summarize'
    }
    response = requests.get(
        f"{BASE_URL}/rag/query", 
        params=params,
        headers=HEADERS
    )
    assert response.status_code == 200
    assert 'answer' in response.json()

def test_list_collections():
    # Test listing collections
    response = requests.get(
        f"{BASE_URL}/rag/collections",
        headers=HEADERS
    )
    assert response.status_code == 200
    collections = response.json()
    assert isinstance(collections, list)
    assert 'test_collection_smart' in collections
    assert 'test_collection_low' in collections

def test_get_info():
    # Test getting system info
    response = requests.get(
        f"{BASE_URL}/rag/info",
        headers=HEADERS
    )
    assert response.status_code == 200
    info = response.json()
    assert isinstance(info, dict)
    assert 'version' in info

def test_unauthorized_access():
    # Test accessing endpoint without token
    response = requests.get(f"{BASE_URL}/rag/info")
    assert response.status_code == 401

    # Test accessing endpoint with wrong token
    wrong_headers = {"Authorization": "Bearer wrong_token"}
    response = requests.get(
        f"{BASE_URL}/rag/info",
        headers=wrong_headers
    )
    assert response.status_code == 401

@pytest.fixture(autouse=True)
def cleanup():
    yield
    # Here you could add cleanup code if needed
    # For example, deleting test collections after tests 