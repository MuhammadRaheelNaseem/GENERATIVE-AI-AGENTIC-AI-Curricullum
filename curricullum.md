# **GENERATIVE AI & AGENTIC AI - COMPLETE CURRICULUM**
## **Maximum Practical | Zero to Production Builder**

---

## **📊 COURSE STRUCTURE**

**Duration:** 26 Weeks  
**Total Projects:** 26 (1 per week)  
**Final Capstone:** 1 Major Multi-Feature System  
**Teaching Model:** 70% Hands-on Building, 30% Concept Learning  

---

## **🎯 LEARNING OUTCOMES**

By the end of this course, students will be able to:
1. Build production-ready LLM applications from scratch
2. Implement RAG systems with citations and quality control
3. Create autonomous AI agents with safety guardrails
4. Design multi-agent systems for complex workflows
5. Validate, test, and deploy AI systems
6. Implement security best practices (OWASP LLM Top 10)
7. Track costs, monitor performance, and optimize systems
8. Switch between multiple LLM providers (OpenAI, Gemini, Grok)
9. Work with both cloud and local models
10. Ship professional AI products with proper documentation

---

## **🛠️ COMPLETE TECHNOLOGY STACK**

### **Programming & Environment**
- Python 3.10+
- Jupyter Notebooks / Google Colab
- VS Code / PyCharm
- Git & GitHub
- Virtual Environments (venv)

### **LLM Providers**
- OpenAI (GPT-4, GPT-4o, GPT-4o-mini)
- Google Gemini (1.5 Pro, 1.5 Flash, 2.0)
- xAI Grok
- Anthropic Claude (API)
- Local Models via Ollama (Llama, Mistral, Phi)

### **Core AI Frameworks**
- LangChain (v0.1+)
- LlamaIndex (v0.9+)
- LangGraph
- CrewAI
- AutoGen (Microsoft)

### **Data Processing**
- Pandas
- NumPy (basics)
- JSON/CSV handling
- PDF parsing (PyPDF2, pdfplumber)
- Document loaders (Unstructured, docx)

### **Vector Databases**
- Qdrant
- Pinecone
- Chroma
- FAISS
- Weaviate

### **Embeddings**
- OpenAI Embeddings
- HuggingFace Embeddings
- Sentence Transformers
- Cohere Embeddings

### **Validation & Quality**
- Pydantic v2
- Instructor
- JSON Schema
- Custom validators
- Regex patterns

### **Web & API**
- Streamlit
- Gradio
- FastAPI
- Flask (basics)
- Requests library
- httpx

### **Testing & Evaluation**
- pytest
- unittest
- Custom eval harnesses
- RAGAS (RAG evaluation)
- DeepEval
- TruLens

### **Monitoring & Observability**
- LangSmith
- Langfuse
- Weights & Biases (W&B)
- Helicone
- Custom logging

### **Security**
- python-dotenv
- OWASP LLM guidelines
- Prompt injection detection
- Input sanitization
- Rate limiting

### **Database**
- SQLite
- PostgreSQL (basics)
- Redis (caching)

### **Deployment**
- Docker basics
- Railway / Render
- Streamlit Cloud
- Vercel (for Next.js if needed)

---

## **📅 DETAILED WEEK-BY-WEEK CURRICULUM**

---

## **🔰 PHASE 0: FOUNDATIONS & SETUP (Weeks 1-3)**

### **Week 1: Python Essentials for AI Development**

**Core Topics:**
- Python setup and environment configuration
- Variables, data types, and operators
- Strings and string manipulation
- Lists, tuples, and basic data structures
- File I/O operations (read, write, append)
- Basic error handling (try-except)

**Practical Focus:**
- Working with text files
- Data cleaning and preprocessing
- JSON file handling
- CSV operations
- Building reusable functions

**Tools Introduced:**
- VS Code setup
- Jupyter Notebooks
- Google Colab
- Git basics (clone, commit, push)

**Build Project:**
**"Text Processing Toolkit"**
- Text file cleaner (remove whitespace, duplicates)
- Character and word counter
- Text splitter (by lines, words, characters)
- JSON converter (text → structured data)
- CSV exporter
- Basic statistics generator

**Deliverables:**
- Working Python toolkit (5+ functions)
- GitHub repository
- README with usage instructions
- Sample input/output files
- 3-minute demo video

---

### **Week 2: APIs, HTTP & Cost Management**

**Core Topics:**
- What is an API and how it works
- HTTP methods (GET, POST)
- API authentication (API keys, tokens)
- Environment variables and security
- Request/Response cycle
- Status codes and error handling
- Rate limiting and quotas
- Token-based pricing models

**Practical Focus:**
- Setting up API keys securely
- Making first API calls
- Parsing API responses
- Error handling patterns
- Building cost calculators
- Tracking API usage

**Tools Introduced:**
- requests library
- python-dotenv
- OpenAI Python SDK
- Postman (API testing)
- curl basics

**Build Project:**
**"Smart API Manager & Cost Tracker"**
- Secure API key loader
- Multi-provider API caller (OpenAI, Gemini)
- Response parser and saver
- Token counter
- Cost calculator (per request)
- Daily/weekly usage tracker
- Budget alert system
- API call logger with timestamps

**Deliverables:**
- API management tool
- Cost tracking dashboard (JSON/CSV)
- 10 successful API call logs
- Error handling examples
- Usage report with cost breakdown

---

### **Week 3: LLM Fundamentals & Parameters**

**Core Topics:**
- How LLMs work (high-level overview)
- Tokens and tokenization
- Model parameters explained:
  - Temperature (0.0 - 2.0)
  - max_tokens
  - top_p (nucleus sampling)
  - frequency_penalty
  - presence_penalty
  - stop sequences
- System vs User vs Assistant messages
- Chat completion vs Text completion
- Streaming responses
- Model selection criteria

**Practical Focus:**
- Parameter experimentation
- Building conversation memory
- System prompt engineering
- Role-based AI personalities
- Context window management
- Response quality optimization

**Tools Introduced:**
- OpenAI Playground
- Token counter tools
- Parameter testing frameworks

**Build Project:**
**"AI Parameter Laboratory"**
- Temperature tester (same prompt, 5 temperatures)
- Parameter comparison tool
- AI personality creator (5 different personas)
- Conversation memory manager
- Context window optimizer
- Response quality scorer
- Parameter recommendation engine

**Deliverables:**
- Parameter testing report (charts/graphs)
- 5 AI personalities with examples
- Chat history manager
- Best practices document
- Comparison spreadsheet

---

## **🎨 PHASE 1: PROMPT ENGINEERING & RELIABILITY (Weeks 4-7)**

### **Week 4: Advanced Prompt Engineering**

**Core Topics:**
- Prompt structure and anatomy
- Role-Context-Task-Format framework
- Zero-shot vs Few-shot prompting
- Chain-of-Thought (CoT) prompting
- Self-consistency techniques
- Prompt templates and variables
- Output format specification
- Constraints and boundaries

**Practical Focus:**
- Writing effective prompts
- Prompt versioning and iteration
- A/B testing prompts
- Building prompt libraries
- Documenting prompt strategies
- Measuring prompt effectiveness

**Tools Introduced:**
- Prompt template engines
- Version control for prompts
- Prompt testing frameworks

**Build Project:**
**"Prompt Engineering Toolkit"**
- Prompt template library (10+ templates)
- Few-shot example generator
- Prompt version controller
- A/B testing framework
- Output format validator
- Prompt quality scorer
- Best prompt selector (based on evals)

**Deliverables:**
- 20+ tested prompt templates
- Prompt versioning system
- Comparison report (v1 vs v2 vs v3)
- Prompt documentation guide
- Quality benchmark results

---

### **Week 5: Hallucination Detection & Reliability**

**Core Topics:**
- What causes hallucinations
- Types of hallucinations (factual, logical, fabricated)
- Reliability patterns:
  - "I don't know" policy
  - Clarifying questions
  - Source citation
  - Confidence scoring
  - Verification loops
- Grounding techniques
- Fact-checking strategies
- Uncertainty quantification

**Practical Focus:**
- Building hallucination detectors
- Implementing refusal logic
- Creating safety rails
- Testing edge cases
- Measuring reliability metrics
- Documenting failure modes

**Tools Introduced:**
- Hallucination detection libraries
- Fact-checking APIs
- Confidence scoring tools

**Build Project:**
**"AI Truth Detector & Safety System"**
- Hallucination classifier
- Uncertainty scorer
- "I don't know" trigger system
- Source citation enforcer
- Clarifying question generator
- Fact verification checker
- Reliability dashboard
- Edge case test suite (50+ questions)

**Deliverables:**
- Working truth detector
- 100-question test set (with expected behaviors)
- Reliability metrics report
- Failure analysis document
- Safety checklist

---

### **Week 6: Multi-Provider Integration**

**Core Topics:**
- Provider comparison (OpenAI, Gemini, Grok, Claude)
- API differences and similarities
- Provider abstraction patterns
- Fallback strategies
- Cost optimization across providers
- Quality benchmarking
- Latency comparison
- Model capability matrix

**Practical Focus:**
- Building provider-agnostic code
- Switching between providers
- Cost-quality tradeoffs
- Provider selection logic
- Unified response format
- Error handling per provider

**Tools Introduced:**
- OpenAI SDK
- Google Generative AI SDK
- Anthropic SDK
- Provider wrapper libraries

**Build Project:**
**"Multi-Provider AI Router"**
- Unified interface for 4 providers
- Provider switcher (config-based)
- Cost comparison tool
- Quality benchmarker
- Latency tester
- Provider recommendation engine
- Automatic fallback system
- Response normalizer

**Deliverables:**
- Working multi-provider app
- Provider comparison report
- Cost analysis spreadsheet
- Quality benchmark (50 prompts × 4 providers)
- Recommendation guide

---

### **Week 7: Function Calling & Tool Use**

**Core Topics:**
- Function calling concepts
- Tool schemas (JSON Schema)
- Tool selection logic
- Argument extraction
- Function execution patterns
- Tool chaining
- Error handling for tools
- Safety constraints

**Practical Focus:**
- Writing tool schemas
- Building safe tools
- Tool allowlisting
- Input validation
- Output sanitization
- Tool call logging
- Building tool libraries

**Tools Introduced:**
- OpenAI function calling
- Gemini function calling
- Pydantic for schemas
- JSON Schema validators

**Build Project:**
**"Smart Assistant with Tool Calling"**
- Calculator tool
- Web search tool (controlled)
- File lookup tool
- Weather tool (API)
- Unit converter
- Tool allowlist manager
- Argument validator
- Tool call logger
- Safety checker (prevent dangerous calls)

**Deliverables:**
- Working tool-using assistant
- 5+ safe tools
- Tool schema library
- Safety test results
- Tool call audit log

**📦 Mini Project 1:** 
**"Smart Study Buddy"** (Combining Weeks 4-7)
- Chat interface with memory
- Document summarization
- Quiz generator (from content)
- Hallucination detection
- Multi-provider support
- Tool calling for calculations
- Cost tracking
- Quality metrics

---

## **🔧 PHASE 2: STRUCTURED OUTPUTS & VALIDATION (Weeks 8-10)**

### **Week 8: Pydantic Deep Dive (Part 1)**

**Core Topics:**
- Why structured data matters
- Pydantic BaseModel basics
- Field types and validators
- Optional vs Required fields
- Default values
- Nested models
- Lists and dictionaries
- Enums and Literals
- Custom validation functions

**Practical Focus:**
- Building data models
- Input validation
- Error message customization
- Type safety
- Schema generation
- JSON serialization
- Model inheritance

**Tools Introduced:**
- Pydantic v2
- Type hints
- dataclasses (comparison)

**Build Project:**
**"Data Validation API"**
- User profile model
- Address validator
- Email/phone validators
- Date range validator
- File upload model
- Nested order system (customer → order → items)
- Custom business rule validators
- Validation error reporter

**Deliverables:**
- 10+ Pydantic models
- Validation test suite
- Error handling examples
- API endpoint demo
- Schema documentation

---

### **Week 9: Pydantic Deep Dive (Part 2) + LLM Integration**

**Core Topics:**
- Pydantic with LLMs
- Structured output enforcement
- Response parsing strategies
- Retry logic on validation failure
- Partial parsing (graceful degradation)
- Confidence fields
- Model configuration
- Serialization options
- Schema-guided generation

**Practical Focus:**
- LLM → Pydantic pipeline
- Handling malformed outputs
- Building robust parsers
- Validation retry loops
- Quality scoring
- Format enforcement

**Tools Introduced:**
- Instructor library
- LLM schema enforcement
- JSON repair tools

**Build Project:**
**"Invoice/Receipt Extractor"**
- Receipt image → OCR text
- Text → structured JSON (Pydantic)
- Fields: vendor, date, items, amounts, tax, total
- Validation rules (total = sum(items) + tax)
- Retry on failure (max 3 attempts)
- Partial extraction fallback
- Confidence scorer
- CSV exporter
- Extraction quality report

**Deliverables:**
- Working extractor (10+ sample receipts)
- Pydantic models
- Validation pass/fail report
- Retry statistics
- Quality metrics

---

### **Week 10: Real-World Data Extraction**

**Core Topics:**
- Information extraction patterns
- Entity recognition
- Relationship extraction
- Multi-document processing
- Batch processing strategies
- Quality assurance
- Human-in-the-loop validation
- Error correction workflows

**Practical Focus:**
- Building extraction pipelines
- Handling variations in format
- Managing extraction quality
- Building validation UIs
- Creating correction workflows
- Measuring accuracy

**Tools Introduced:**
- PDF parsers (pdfplumber, PyPDF2)
- OCR tools (Tesseract, cloud OCR)
- NER libraries (spaCy basics)

**Build Project:**
**"Resume/CV Parser & Scorer"**
- Parse PDF/DOCX resumes
- Extract:
  - Personal info (name, email, phone)
  - Education (degrees, institutions, dates)
  - Experience (companies, roles, dates, descriptions)
  - Skills (technical, soft)
  - Certifications
- Validate extracted data
- Score completeness (0-100%)
- Generate structured JSON
- Export to CSV/Excel
- Flag missing critical fields

**Deliverables:**
- Resume parser (20+ sample CVs)
- Pydantic models for CV data
- Extraction accuracy report
- Scoring algorithm
- Validation rules
- Sample outputs (JSON, CSV)

**📦 Mini Project 2:**
**"Job Application Processor"**
- Parse resumes
- Match against job requirements
- Score candidates (0-100%)
- Extract key qualifications
- Generate interview questions
- Create comparison report

---

## **🤖 PHASE 3: OPEN-SOURCE & LOCAL MODELS (Weeks 11-12)**

### **Week 11: Local & Open-Source Models**

**Core Topics:**
- Open-source vs Closed-source models
- HuggingFace ecosystem
- Model selection criteria (size, quality, speed)
- Quantization (4-bit, 8-bit)
- Running models locally (CPU vs GPU)
- Ollama setup and usage
- Model comparison (Llama, Mistral, Phi, Gemma)
- Privacy and compliance benefits
- Offline AI capabilities

**Practical Focus:**
- Installing Ollama
- Downloading models
- Running inference locally
- Comparing cloud vs local
- Measuring latency
- Cost savings calculation
- Building offline apps

**Tools Introduced:**
- Ollama
- HuggingFace Hub
- llama.cpp basics
- Transformers library (minimal)

**Build Project:**
**"Offline AI Assistant"**
- Local chatbot (Ollama)
- Document Q&A (offline)
- Text summarization
- Translation
- Code helper
- Latency comparison (local vs API)
- Cost comparison report
- Privacy compliance checker

**Deliverables:**
- Working offline assistant
- 5 local models tested
- Performance comparison (speed, quality)
- Cost analysis (local vs cloud)
- Privacy compliance report
- Setup guide

---

### **Week 12: Model Evaluation & Selection**

**Core Topics:**
- Evaluation frameworks
- Accuracy metrics
- Format compliance scoring
- Factuality testing
- Consistency testing
- Cost-quality tradeoffs
- Latency benchmarking
- Model leaderboards
- Custom evaluation sets

**Practical Focus:**
- Building eval datasets
- Running systematic tests
- Scoring responses
- Creating leaderboards
- Prompt portability testing
- Provider migration strategies

**Tools Introduced:**
- RAGAS (for RAG eval)
- DeepEval
- Custom eval harnesses
- Benchmark datasets

**Build Project:**
**"Model Selection Framework"**
- Evaluation dataset (100+ questions)
- Multi-model tester (OpenAI, Gemini, Grok, local)
- Automated scoring system:
  - Format compliance (0-100%)
  - Factual accuracy (0-100%)
  - Consistency score
  - Refusal appropriateness
  - Cost per query
  - Latency (ms)
- Leaderboard generator
- Model recommendation engine
- Migration guide generator

**Deliverables:**
- 100-question eval set
- 5+ models tested
- Leaderboard report
- Best model recommendations (by use case)
- Migration playbook

**📦 Mini Project 3:**
**"Company Knowledge Helper"** (Local + Private)
- Offline chatbot
- Company docs search
- Privacy-first design
- No data leaves device
- Multi-model support
- Quality benchmarking

---

## **📚 PHASE 4: RAG SYSTEMS (Weeks 13-17)**

### **Week 13: Embeddings & Semantic Search**

**Core Topics:**
- What are embeddings (vector representations)
- Embedding models (OpenAI, HuggingFace, Cohere)
- Vector similarity (cosine, dot product, euclidean)
- Semantic search vs keyword search
- Chunking strategies:
  - Fixed-size chunking
  - Sentence-based chunking
  - Paragraph-based chunking
  - Semantic chunking
  - Sliding window
- Chunk size optimization
- Overlap techniques
- Metadata preservation

**Practical Focus:**
- Generating embeddings
- Computing similarity
- Building search engines
- Chunking documents
- Testing chunk strategies
- Measuring search quality

**Tools Introduced:**
- OpenAI Embeddings API
- sentence-transformers
- scikit-learn (cosine similarity)
- LangChain text splitters

**Build Project:**
**"Semantic Search Engine for Documents"**
- Document uploader (PDF, TXT, DOCX)
- Chunking with 3 strategies
- Embed all chunks
- Semantic search interface
- Top-K retrieval
- Similarity score display
- Chunk visualization
- Chunking strategy comparison report

**Deliverables:**
- Working search engine
- 20+ documents indexed
- Chunking comparison (which worked best?)
- Search quality metrics
- Sample queries + results

---

### **Week 14: Vector Databases**

**Core Topics:**
- Why vector databases?
- Vector DB comparison (Qdrant, Pinecone, Chroma, FAISS)
- Indexing strategies
- Metadata filtering
- Hybrid search (vector + keyword)
- CRUD operations (Create, Read, Update, Delete)
- Collections and namespaces
- Persistence vs in-memory
- Scaling considerations

**Practical Focus:**
- Setting up vector DBs
- Indexing documents
- Searching with filters
- Updating indexes
- Managing metadata
- Building hybrid search

**Tools Introduced:**
- Qdrant (local + cloud)
- Pinecone
- Chroma
- FAISS
- LlamaIndex integration

**Build Project:**
**"Smart Document Library with Filters"**
- Upload and index documents
- Metadata fields:
  - Document type (policy, note, article)
  - Date
  - Author
  - Department
  - Tags
- Search with filters (e.g., "only policies from 2024")
- Update/delete documents
- Collection management
- Version tracking
- Backup/restore functionality

**Deliverables:**
- Working document library
- 50+ documents indexed
- Filter query examples
- CRUD operation demos
- Performance report (indexing time, search speed)

---

### **Week 15: RAG Pipeline - Basic**

**Core Topics:**
- RAG architecture (Retrieval + Generation)
- Query understanding
- Retrieval strategies
- Context construction
- Prompt engineering for RAG
- Response generation
- Citation formatting
- "Answer only from sources" policy
- Handling "no relevant docs" cases

**Practical Focus:**
- Building end-to-end RAG
- Retrieval quality
- Context relevance
- Citation extraction
- Failure handling
- Quality measurement

**Tools Introduced:**
- LangChain RAG
- LlamaIndex
- Haystack (optional)

**Build Project:**
**"PDF Chat with Mandatory Citations"**
- Upload PDFs
- Chunk and index
- Chat interface
- Retrieve relevant chunks
- Generate answer with citations
- Citation format: [Source: filename, page X]
- "I cannot find relevant information" fallback
- Source highlighting
- Confidence scoring

**Deliverables:**
- Working PDF chat
- 10+ PDFs tested
- Citation examples
- Failure case handling
- User guide

---

### **Week 16: RAG Quality Improvement**

**Core Topics:**
- Common RAG failures:
  - Wrong chunk retrieved
  - Missing relevant chunks
  - Poor context quality
  - Hallucination despite sources
  - Over-reliance on single chunk
- Reranking techniques
- Hybrid search (vector + BM25)
- Query expansion
- HyDE (Hypothetical Document Embeddings)
- Parent-child chunking
- Context compression
- Multi-query retrieval

**Practical Focus:**
- Building evaluation sets
- Measuring retrieval accuracy
- Testing improvements
- A/B testing RAG versions
- Documenting wins/losses

**Tools Introduced:**
- Cohere Rerank
- BM25 algorithms
- Query expansion tools
- LlamaIndex advanced RAG

**Build Project:**
**"RAG Quality Upgrade Sprint"**
- Create evaluation dataset (50 Q&A pairs)
- Baseline RAG (v1)
- Implement improvements:
  - Reranking
  - Hybrid search
  - Query expansion
  - Better chunking
- Measure improvements:
  - Answer accuracy (v1 vs v2)
  - Citation accuracy
  - Response time
  - Cost per query
- A/B comparison report

**Deliverables:**
- RAG v1 and v2 systems
- 50-question eval set with ground truth
- Before/after metrics
- Improvement recommendations
- Cost-quality analysis

---

### **Week 17: Production RAG**

**Core Topics:**
- RAG at scale
- Caching strategies
- Rate limiting
- Error handling and retries
- Monitoring and logging
- User feedback loops
- Continuous improvement
- Security (prompt injection in queries)
- Compliance and data governance

**Practical Focus:**
- Building production-ready RAG
- Adding observability
- Implementing feedback
- Security hardening
- Cost optimization
- Performance tuning

**Tools Introduced:**
- Redis (caching)
- LangSmith (monitoring)
- Langfuse
- Custom logging

**Build Project:**
**"Enterprise Knowledge Base System"**
- Production RAG with:
  - Caching (frequently asked questions)
  - Rate limiting (per user)
  - Request logging
  - Error tracking
  - User feedback (thumbs up/down)
  - Admin dashboard
  - Usage analytics
  - Cost monitoring
  - Security checks (prompt injection detection)
  - API documentation

**Deliverables:**
- Production RAG system
- Admin dashboard
- Monitoring setup
- Security test report
- API documentation
- Performance metrics

**📦 Mini Project 4:**
**"Institute Policy Bot"**
- RAG on institute policies
- Department-specific answers
- Mandatory citations
- Feedback collection
- Quality monitoring
- Admin review queue

---

## **⚙️ PHASE 5: ORCHESTRATION & DEPLOYMENT (Weeks 18-20)**

### **Week 18: LangChain/LlamaIndex Pipelines**

**Core Topics:**
- Chain concepts (sequential, parallel)
- LCEL (LangChain Expression Language)
- Prompt chains
- Document processing chains
- Router chains
- Map-reduce patterns
- Stuff, Refine, Map-rerank strategies
- Error handling in chains
- Chain debugging

**Practical Focus:**
- Building complex pipelines
- Chaining operations
- Parallel processing
- Error recovery
- Testing chains
- Performance optimization

**Tools Introduced:**
- LangChain LCEL
- LlamaIndex pipelines
- Chain debugger

**Build Project:**
**"Document Intelligence Pipeline"**
- Upload long document
- Pipeline stages:
  1. Split into sections
  2. Summarize each section
  3. Extract key entities
  4. Generate action items
  5. Create executive summary
- Progress tracking
- Intermediate results viewer
- Error recovery
- Cost tracking per stage

**Deliverables:**
- Working pipeline
- 10+ documents processed
- Stage-wise outputs
- Performance metrics
- Error handling examples

---

### **Week 19: Production Patterns & Best Practices**

**Core Topics:**
- Production checklist
- Configuration management
- Environment separation (dev, staging, prod)
- Secrets management
- Logging best practices
- Error monitoring
- Health checks
- Graceful degradation
- Circuit breakers
- Retry strategies with exponential backoff

**Practical Focus:**
- Building production-ready apps
- Adding observability
- Implementing reliability patterns
- Security hardening
- Documentation

**Tools Introduced:**
- Logging frameworks (loguru)
- Sentry (error tracking)
- Health check patterns
- Config management (pydantic-settings)

**Build Project:**
**"Meeting Minutes → Action Items System"**
- Input: Meeting transcript
- Output: Structured action items
  - Task description
  - Assigned to (extracted from transcript)
  - Deadline (extracted or inferred)
  - Priority (high/medium/low)
  - Dependencies
- Validation rules
- Retry logic
- Audit logging
- Email/Slack notification stubs
- Admin review interface

**Deliverables:**
- Production-ready system
- 20+ sample transcripts processed
- Action item database
- Audit logs
- Error handling examples
- Admin documentation

---

### **Week 20: Deployment & APIs**

**Core Topics:**
- FastAPI basics
- REST API design
- Request validation
- Response models
- Authentication (API keys, JWT basics)
- Rate limiting
- CORS configuration
- API documentation (OpenAPI/Swagger)
- Deployment options:
  - Local deployment
  - Cloud deployment (Railway, Render, Fly.io)
  - Containerization basics (Docker intro)

**Practical Focus:**
- Building REST APIs
- Adding authentication
- Implementing rate limits
- Writing API docs
- Deploying to cloud
- Testing APIs

**Tools Introduced:**
- FastAPI
- Uvicorn
- Railway/Render
- Docker basics
- API testing (httpie, curl)

**Build Project:**
**"Deployed RAG API with Authentication"**
- RESTful API endpoints:
  - POST /upload (upload documents)
  - POST /query (ask questions)
  - GET /documents (list indexed docs)
  - DELETE /documents/{id}
- API key authentication
- Rate limiting (10 requests/minute)
- Request logging
- API documentation (auto-generated)
- Cloud deployment
- Health check endpoint
- Usage analytics endpoint

**Deliverables:**
- Deployed API (live URL)
- API documentation
- Client examples (Python, cURL)
- Rate limiting demo
- Deployment guide

**📦 Mini Project 5:**
**"Research Assistant API"**
- Multi-document synthesis
- Citation extraction
- Structured outline generation
- Export to multiple formats
- API access
- Authentication
- Usage tracking

---

## **🤖 PHASE 6: AI AGENTS (Weeks 21-24)**

### **Week 21: Introduction to Agents**

**Core Topics:**
- Agent vs Chain (key differences)
- Agent architecture (ReAct pattern)
- Reasoning and Acting
- Tool selection logic
- Observation and thought loops
- Stopping conditions
- Agent types:
  - Zero-shot React
  - Conversational
  - Plan-and-execute
- Agent limitations and failures

**Practical Focus:**
- Building simple agents
- Tool design for agents
- Stopping criteria
- Error handling
- Logging agent decisions
- Testing agent behavior

**Tools Introduced:**
- LangChain Agents
- LlamaIndex Agents
- Agent executors

**Build Project:**
**"Task Runner Agent"**
- Tools available:
  - Calculator
  - Text summarizer
  - Web search (controlled/stub)
  - File reader
  - Data extractor
- Agent decides which tool(s) to use
- Thought process logging
- Max iterations limit (prevent infinite loops)
- Stop on success or max iterations
- Tool call audit log
- Failure case analysis

**Deliverables:**
- Working agent
- 20+ task examples
- Decision logs
- Failure analysis
- Tool usage statistics

---

### **Week 22: Agent Security & Safety**

**Core Topics:**
- Agent risks and vulnerabilities
- Prompt injection in agents
- Tool allowlisting
- Input sanitization
- Output validation
- Tool permission systems
- Human-in-the-loop patterns
- Approval gates
- Sandbox execution
- Agent observability

**Practical Focus:**
- Securing agent tools
- Building approval workflows
- Testing security
- Preventing misuse
- Logging all actions
- Building kill switches

**Tools Introduced:**
- Tool validators
- Approval UI patterns
- Security testing tools

**Build Project:**
**"Safe Research Agent with Guardrails"**
- Research agent with tools:
  - Web search (allowlisted domains only)
  - Document fetcher (URL validator)
  - Summarizer
  - Citation extractor
- Security features:
  - Domain allowlist (only .edu, .gov, approved sites)
  - URL validator (no suspicious patterns)
  - Output sanitizer
  - Prompt injection detector
  - Tool call approvals (for sensitive operations)
  - Rate limits per tool
  - Audit logging
- Security test suite (attempted attacks)

**Deliverables:**
- Secure agent
- Domain allowlist
- Security test results (10+ attack attempts)
- Audit logs
- Security documentation

---

### **Week 23: Agent Memory Systems**

**Core Topics:**
- Short-term vs long-term memory
- Conversation memory
- Entity memory
- Episodic memory
- Semantic memory
- Memory retrieval strategies
- Memory summarization
- Memory persistence
- Forgetting mechanisms
- Personalization with memory

**Practical Focus:**
- Building memory stores
- Implementing retrieval
- Managing memory size
- Personalizing responses
- Testing memory quality

**Tools Introduced:**
- LangChain memory
- Vector stores for memory
- Memory summarizers

**Build Project:**
**"Personal Tutor Agent with Student Profile"**
- Student profile memory:
  - Learning goals
  - Current level (beginner/intermediate/advanced)
  - Weak areas
  - Preferred learning style
  - Past questions and performance
- Agent capabilities:
  - Personalized explanations
  - Difficulty adaptation
  - Progress tracking
  - Quiz generation (based on weak areas)
  - Study plan creation
- Memory management:
  - Session memory (current conversation)
  - Long-term profile (persisted)
  - Performance history
  - Concept graph (what student knows)

**Deliverables:**
- Tutor agent with memory
- Student profile system
- 5 sample student profiles
- Progress tracking
- Personalization examples

---

### **Week 24: Advanced Agent Patterns**

**Core Topics:**
- Reflexion (self-critique and improvement)
- Plan-and-execute agents
- Multi-step reasoning
- Self-correction
- Tool creation by agents
- Meta-prompting
- Agent debugging techniques
- Performance optimization
- Cost optimization for agents

**Practical Focus:**
- Building self-correcting agents
- Planning systems
- Optimization techniques
- Complex workflows
- Testing advanced patterns

**Tools Introduced:**
- Advanced LangChain patterns
- Custom agent executors
- Agent monitoring tools

**Build Project:**
**"Self-Improving Research Agent"**
- Research workflow:
  1. Understand query
  2. Plan research steps
  3. Execute searches
  4. Synthesize findings
  5. Self-critique answer
  6. Improve if needed (max 2 iterations)
- Features:
  - Planning visualization
  - Step-by-step execution logs
  - Self-critique mechanism
  - Quality threshold (0.8/1.0)
  - Auto-improvement loop
  - Final report generation
  - Citation verification

**Deliverables:**
- Self-improving agent
- 10 research tasks completed
- Planning examples
- Self-critique logs
- Quality improvement metrics

**📦 Mini Project 6:**
**"Web Research → Report Generator"**
- Research query understanding
- Multi-source research
- Information synthesis
- Report structure generation
- Citation management
- Quality assurance
- Export to PDF/DOCX

---

## **👥 PHASE 7: MULTI-AGENT SYSTEMS (Weeks 25-26)**

### **Week 25: CrewAI Multi-Agent Teams**

**Core Topics:**
- Multi-agent architecture
- Role-based agents
- Task delegation
- Agent collaboration
- Hierarchical vs Sequential teams
- Communication protocols
- Conflict resolution
- Crew orchestration
- Task dependencies
- Acceptance criteria

**Practical Focus:**
- Designing agent teams
- Role definition
- Task breakdown
- Coordination patterns
- Testing team performance
- Managing agent communication

**Tools Introduced:**
- CrewAI
- Agent role design
- Task orchestration

**Build Project:**
**"Content Creation Team"**
- Agent roles:
  1. Researcher (finds information, creates source list)
  2. Writer (creates first draft)
  3. Editor (improves clarity, structure, grammar)
  4. Fact-checker (verifies claims against sources)
  5. SEO Optimizer (adds keywords, meta descriptions)
- Workflow:
  - Researcher → Writer → Editor → Fact-checker → SEO
- Features:
  - Role-specific prompts
  - Task handoffs with context
  - Revision tracking
  - Quality gates (fact-check pass/fail)
  - Final output: Blog post + source list + SEO metadata

**Deliverables:**
- Working multi-agent team
- 5 blog posts created
- Workflow visualization
- Agent interaction logs
- Quality metrics per agent

---

### **Week 26: LangGraph Workflows & Human-in-the-Loop**

**Core Topics:**
- Graph-based workflows
- State machines
- Conditional routing
- Parallel execution
- Checkpointing and persistence
- Human approval gates
- Rollback mechanisms
- Workflow visualization
- State inspection
- Error recovery in graphs

**Practical Focus:**
- Building stateful workflows
- Adding human approvals
- Implementing rollback
- Visualizing workflows
- Testing complex flows
- State persistence

**Tools Introduced:**
- LangGraph
- State management
- Workflow visualization
- Checkpoint systems

**Build Project:**
**"Customer Support Multi-Agent System"**
- Agents:
  1. Intake Agent (categorize ticket)
  2. Triage Agent (assess severity)
  3. Routing Agent (assign to specialist)
  4. Specialist Agents:
     - Technical Support
     - Billing Support
     - Product Support
  5. Quality Assurance Agent
  6. Escalation Agent
- Workflow features:
  - State tracking (ticket status)
  - Human approval for refunds >$100
  - Escalation path (unresolved after 2 attempts)
  - Persistent state (can pause/resume)
  - Audit logging
  - Performance dashboard

**Deliverables:**
- Multi-agent support system
- 50 simulated tickets processed
- Workflow graph visualization
- State transition logs
- Performance metrics (resolution time, escalation rate)
- Human approval examples

---

## **🏆 FINAL CAPSTONE PROJECT (Week 27-28)**

### **Institute Office Assistant - Complete System**

**Comprehensive Requirements:**

#### **Core Features**

**1. RAG Knowledge Base**
- Multi-document support (policies, handbooks, FAQs)
- Department-based filtering
- Semantic search + keyword search (hybrid)
- Mandatory citation format: [Source: Document Name, Page/Section]
- "Answer only from uploaded documents" mode
- Confidence scoring (high/medium/low)
- "Cannot find answer" handling

**2. Tool Suite (Controlled)**
- Calculator (math operations)
- Policy lookup (by name/keyword)
- Fee calculator (based on policy rules)
- Calendar tools (academic dates - read-only)
- Form generator:
  - Admission inquiry form
  - Complaint form
  - Leave application
  - Fee query form
- All outputs as validated JSON (Pydantic models)

**3. Multi-Agent Team**
- **Intake Agent**: Categorize query (academic/admin/fees/technical)
- **Department Router**: Route to right specialist
- **Specialist Agents**:
  - Academic Support (courses, schedules, requirements)
  - Administrative Support (policies, procedures)
  - Fee & Finance Support (calculations, payment queries)
  - Technical Support (IT, systems)
- **QA Agent**: Verify answer quality
- **Escalation Agent**: Handle edge cases

**4. Structured Outputs (Pydantic Models)**
```
AdmissionInquiry:
  - student_name
  - contact_info
  - program_of_interest
  - qualification
  - query_details
  - urgency_level

StudentProfile:
  - student_id
  - name
  - department
  - semester
  - contact_info
  - status

ComplaintTicket:
  - ticket_id
  - category
  - priority
  - description
  - reporter
  - timestamp
  - status

FeeQuery:
  - student_id
  - query_type
  - amount
  - semester
  - response
```

**5. Security & Safety**
- Prompt injection detection and blocking
- Tool allowlist (no unauthorized tools)
- Input sanitization
- Output validation
- Refusal policy (cannot answer personal medical/legal queries)
- Rate limiting (per user)
- Audit logging (all actions logged)
- Data privacy compliance

**6. Evaluation & Monitoring**
- 100-question evaluation set:
  - 40 policy questions
  - 20 calculation questions
  - 20 procedural questions
  - 10 edge cases
  - 10 should-refuse questions
- Metrics:
  - Answer accuracy (0-100%)
  - Citation accuracy
  - Refusal appropriateness
  - Response time
  - Cost per query
  - User satisfaction (thumbs up/down)
- Dashboard:
  - Daily query volume
  - Category breakdown
  - Agent performance
  - Cost tracking
  - Error rates

**7. User Interface**
- Streamlit web app:
  - Chat interface
  - Document upload (admin)
  - Knowledge base viewer
  - Form generation UI
  - Admin dashboard
  - Evaluation results
  - Audit logs viewer

**8. Deployment**
- Containerized (Docker)
- Deployed to cloud (Railway/Render)
- API endpoints
- Authentication (admin vs user)
- Health checks
- Monitoring

#### **Technical Stack Requirements**
- Python 3.10+
- LangChain + LangGraph
- CrewAI
- Pydantic v2
- Vector DB (Qdrant/Pinecone)
- FastAPI + Streamlit
- LangSmith/Langfuse (monitoring)
- Docker
- Git/GitHub

#### **Deliverables**

**1. GitHub Repository**
- Well-organized code
- README with:
  - Project overview
  - Features list
  - Setup instructions
  - API documentation
  - Architecture diagram
  - Screenshots
- requirements.txt
- .env.example
- Docker files

**2. Demo Video (10-15 minutes)**
- System overview
- Live demo of key features:
  - RAG query with citation
  - Tool usage (calculator, form generator)
  - Multi-agent workflow
  - Security features (prompt injection blocking)
  - Admin dashboard
- Walkthrough of architecture

**3. Evaluation Report**
- 100-question test results
- Before/after metrics (if iterative improvements made)
- Accuracy breakdown by category
- Failure analysis
- Cost analysis
- Performance metrics

**4. Security Test Report**
- Prompt injection attempts (10+)
- Results (blocked/allowed)
- Tool misuse attempts
- Security checklist completion

**5. Deployment Documentation**
- Deployment steps
- Environment variables
- Cloud setup guide
- API documentation
- User guide
- Admin guide

**6. Presentation**
- 15-minute presentation covering:
  - Problem statement
  - Solution architecture
  - Key features
  - Technical challenges and solutions
  - Results and metrics
  - Future improvements

#### **Grading Rubric (100 points)**

| Category | Points | Criteria |
|----------|--------|----------|
| **Functionality** | 30 | All core features work end-to-end |
| **RAG Quality** | 15 | Accurate answers with proper citations |
| **Agent System** | 15 | Multi-agent workflow functions correctly |
| **Security** | 10 | Prompt injection defenses, safe tools |
| **Validation** | 10 | Pydantic models, error handling |
| **Evaluation** | 10 | Comprehensive testing, metrics |
| **Documentation** | 5 | Clear README, API docs, guides |
| **Deployment** | 5 | Successfully deployed and accessible |

---

## **📊 WEEKLY DELIVERABLES TEMPLATE**

Every week, students submit:

1. **Working Code**
   - GitHub repository link
   - All code runs without errors
   - Clear folder structure

2. **README.md**
   - What the project does
   - How to run it
   - Requirements/dependencies
   - Example usage
   - Known limitations

3. **Demo Video (3-5 minutes)**
   - Screen recording
   - Shows key features
   - Explains decisions made
   - Shows edge cases/failures

4. **Evaluation Report** (where applicable)
   - Test cases
   - Pass/fail results
   - Accuracy metrics
   - Cost analysis

5. **Reflection Document**
   - What worked well
   - Challenges faced
   - How challenges were solved
   - Lessons learned
   - What could be improved

---

## **🎓 GRADING STRUCTURE**

| Component | Weight |
|-----------|--------|
| Weekly Projects (26 × 2.5%) | 65% |
| Mini Projects (6 × 3%) | 18% |
| Final Capstone | 15% |
| Participation & Progress | 2% |

**Weekly Project Scoring:**
- Code Quality & Functionality: 40%
- Reliability & Error Handling: 25%
- Documentation: 15%
- Demo Video: 10%
- Evaluation/Testing: 10%

---

## **🛠️ SUPPORT RESOURCES PROVIDED**

1. **Code Templates**
   - Project structure templates
   - README templates
   - Pydantic model templates
   - API endpoint templates
   - Testing templates

2. **Datasets**
   - Sample documents (50+)
   - Sample policies (20+)
   - Sample resumes (30+)
   - Sample receipts (50+)
   - Sample transcripts (20+)

3. **Evaluation Sets**
   - Pre-built test questions
   - Scoring rubrics
   - Benchmark datasets

4. **Checklists**
   - Security checklist
   - Production readiness checklist
   - Code review checklist
   - Deployment checklist

5. **Reference Implementations**
   - Sample mini-projects
   - Common patterns library
   - Best practices guide

---

## **📈 SKILL PROGRESSION TRACKER**

| Week | Python | LLMs | RAG | Agents | Production |
|------|--------|------|-----|--------|------------|
| 1-3 | ███░░ | █░░░░ | ░░░░░ | ░░░░░ | ░░░░░ |
| 4-7 | ████░ | ███░░ | ░░░░░ | ░░░░░ | ░░░░░ |
| 8-10 | ████░ | ████░ | █░░░░ | ░░░░░ | █░░░░ |
| 11-12 | ████░ | ████░ | ██░░░ | ░░░░░ | █░░░░ |
| 13-17 | ████░ | ████░ | █████ | ░░░░░ | ██░░░ |
| 18-20 | ████░ | ████░ | █████ | █░░░░ | ████░ |
| 21-24 | █████ | █████ | █████ | ████░ | ████░ |
| 25-28 | █████ | █████ | █████ | █████ | █████ |

---

## **💼 CAREER-READY PORTFOLIO**

Upon completion, students will have:

1. **26 Working Projects** (GitHub repositories)
2. **6 Mini-Projects** (integrated systems)
3. **1 Capstone Project** (production-grade system)
4. **100+ Hours** of hands-on coding
5. **Real-world experience** with production patterns
6. **Security knowledge** (OWASP LLM Top 10)
7. **Multi-provider expertise** (OpenAI, Gemini, Grok, local models)
8. **Deployment experience** (cloud deployment, APIs)
9. **Evaluation skills** (testing, benchmarking, quality assurance)

**Portfolio Value:**
- Demonstrates end-to-end AI development skills
- Shows progression from basics to advanced
- Includes security and production considerations
- Showcases real-world problem-solving
- Ready for job applications and interviews

---

## **🔄 CONTINUOUS IMPROVEMENT**

Throughout the course:
- Weekly code reviews (peer + instructor)
- Best practices sharing sessions
- Failure analysis discussions
- Tool/library updates as needed
- Real-world case study integrations
- Industry guest sessions (optional)
