from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ai-pdf-chatbot",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Professional AI PDF Chatbot with Ollama",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ai-pdf-chatbot",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "streamlit>=1.28.0",
        "langchain-community>=0.0.10",
        "langchain-text-splitters>=0.0.1",
        "langchain-ollama>=0.1.0",
        "faiss-cpu>=1.7.4",
        "pypdf>=3.17.0",
        "sentence-transformers>=2.2.2",
    ],
    keywords="ai, chatbot, pdf, ollama, langchain, streamlit",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/ai-pdf-chatbot/issues",
        "Documentation": "https://github.com/yourusername/ai-pdf-chatbot#readme",
        "Source Code": "https://github.com/yourusername/ai-pdf-chatbot",
    },
)