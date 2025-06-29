# Research & Content Agent Implementation

## Overview

The Research & Content Agent serves as a foundational component of the A2A agent ecosystem, specializing in market research, keyword analysis, and content generation. This agent combines advanced web research capabilities with sophisticated content creation tools, leveraging Google Cloud's Vertex AI platform to deliver high-quality, SEO-optimized content tailored to specific business requirements.

## Agent Architecture

### Core Agent Class Implementation

```python
import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum

from google.cloud import aiplatform
from google.cloud import pubsub_v1
from google.cloud import storage
from google.cloud import firestore
import vertexai
from vertexai.language_models import TextGenerationModel, ChatModel
from vertexai.generative_models import GenerativeModel

# Import our A2A communication framework
from a2a_communication import AgentCommunicationManager, AgentRegistry

class ContentType(Enum):
    BLOG_POST = "blog_post"
    WEBSITE_COPY = "website_copy"
    SOCIAL_MEDIA = "social_media"
    EMAIL_CAMPAIGN = "email_campaign"
    TECHNICAL_DOCUMENTATION = "technical_documentation"
    MARKETING_COPY = "marketing_copy"

class ResearchScope(Enum):
    COMPETITOR_ANALYSIS = "competitor_analysis"
    KEYWORD_RESEARCH = "keyword_research"
    MARKET_TRENDS = "market_trends"
    AUDIENCE_ANALYSIS = "audience_analysis"
    CONTENT_GAP_ANALYSIS = "content_gap_analysis"

@dataclass
class ResearchRequest:
    """Structured request for research tasks"""
    research_type: ResearchScope
    target_audience: str
    industry_focus: str
    geographic_scope: str
    keywords: List[str]
    competitor_urls: List[str] = None
    content_goals: List[str] = None
    output_format: str = "json"
    priority: str = "normal"
    
@dataclass
class ContentRequest:
    """Structured request for content generation"""
    content_type: ContentType
    topic: str
    target_keywords: List[str]
    audience_persona: str
    tone_of_voice: str
    word_count: int
    seo_requirements: Dict[str, Any]
    brand_guidelines: Dict[str, Any] = None
    reference_materials: List[str] = None
    
@dataclass
class AgentResponse:
    """Standardized response format"""
    success: bool
    data: Dict[str, Any]
    metadata: Dict[str, Any]
    errors: List[str] = None
    processing_time: float = 0.0

class ResearchContentAgent:
    """
    Advanced Research & Content Agent with A2A communication capabilities
    """
    
    def __init__(self, project_id: str, agent_id: str, location: str = "us-central1"):
        self.project_id = project_id
        self.agent_id = agent_id
        self.location = location
        
        # Initialize Google Cloud services
        self.storage_client = storage.Client(project=project_id)
        self.firestore_client = firestore.Client(project=project_id)
        
        # Initialize Vertex AI
        vertexai.init(project=project_id, location=location)
        
        # Initialize A2A communication
        self.comm_manager = AgentCommunicationManager(project_id)
        self.agent_registry = AgentRegistry(self.firestore_client)
        
        # Model configurations with token allocations
        self.model_configs = {
            "research": {
                "model_name": "gemini-1.5-pro",
                "max_tokens": 8192,
                "temperature": 0.3,
                "top_p": 0.8,
                "addon_tokens": 2048  # Additional tokens for add-ons
            },
            "content_generation": {
                "model_name": "gemini-1.5-pro",
                "max_tokens": 8192,
                "temperature": 0.7,
                "top_p": 0.9,
                "addon_tokens": 2048
            },
            "analysis": {
                "model_name": "gemini-1.5-flash",
                "max_tokens": 4096,
                "temperature": 0.2,
                "top_p": 0.8,
                "addon_tokens": 1024
            }
        }
        
        # Tool registry
        self.tools = self._initialize_tools()
        
        # Register agent in the system
        self._register_agent()
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(f"ResearchContentAgent-{agent_id}")
        
    def _initialize_tools(self) -> Dict[str, Any]:
        """Initialize available tools for the agent"""
        return {
            "web_search": WebSearchTool(self.project_id),
            "competitor_analyzer": CompetitorAnalysisTool(self.project_id),
            "keyword_researcher": KeywordResearchTool(self.project_id),
            "content_generator": ContentGenerationTool(self.project_id),
            "seo_optimizer": SEOOptimizationTool(self.project_id),
            "trend_analyzer": TrendAnalysisTool(self.project_id),
            "audience_profiler": AudienceProfilingTool(self.project_id),
            "content_validator": ContentValidationTool(self.project_id)
        }
    
    def _register_agent(self):
        """Register this agent in the agent registry"""
        agent_info = {
            "agent_id": self.agent_id,
            "agent_type": "research_content",
            "capabilities": [
                "market_research",
                "keyword_analysis", 
                "content_generation",
                "competitor_analysis",
                "seo_optimization",
                "trend_analysis"
            ],
            "input_topics": [
                f"agent_{self.agent_id}_requests",
                "research_requests",
                "content_requests"
            ],
            "output_topics": [
                f"agent_{self.agent_id}_responses",
                "research_results",
                "content_deliveries"
            ],
            "metadata": {
                "version": "1.0.0",
                "supported_content_types": [ct.value for ct in ContentType],
                "supported_research_types": [rs.value for rs in ResearchScope]
            }
        }
        
        self.agent_registry.register_agent(agent_info)
        self.logger.info(f"Agent {self.agent_id} registered successfully")

    async def process_message(self, message: Dict[str, Any]) -> AgentResponse:
        """
        Main message processing method for A2A communication
        """
        start_time = datetime.utcnow()
        
        try:
            message_type = message.get("type")
            payload = message.get("payload", {})
            
            if message_type == "research_request":
                result = await self.handle_research_request(payload)
            elif message_type == "content_request":
                result = await self.handle_content_request(payload)
            elif message_type == "analysis_request":
                result = await self.handle_analysis_request(payload)
            else:
                raise ValueError(f"Unknown message type: {message_type}")
                
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            return AgentResponse(
                success=True,
                data=result,
                metadata={
                    "agent_id": self.agent_id,
                    "processing_time": processing_time,
                    "timestamp": datetime.utcnow().isoformat()
                },
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            self.logger.error(f"Error processing message: {str(e)}")
            
            return AgentResponse(
                success=False,
                data={},
                metadata={
                    "agent_id": self.agent_id,
                    "processing_time": processing_time,
                    "timestamp": datetime.utcnow().isoformat()
                },
                errors=[str(e)],
                processing_time=processing_time
            )

    async def handle_research_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle research-specific requests"""
        research_request = ResearchRequest(**payload)
        
        self.logger.info(f"Processing research request: {research_request.research_type}")
        
        if research_request.research_type == ResearchScope.KEYWORD_RESEARCH:
            return await self._perform_keyword_research(research_request)
        elif research_request.research_type == ResearchScope.COMPETITOR_ANALYSIS:
            return await self._perform_competitor_analysis(research_request)
        elif research_request.research_type == ResearchScope.MARKET_TRENDS:
            return await self._analyze_market_trends(research_request)
        elif research_request.research_type == ResearchScope.AUDIENCE_ANALYSIS:
            return await self._analyze_audience(research_request)
        elif research_request.research_type == ResearchScope.CONTENT_GAP_ANALYSIS:
            return await self._analyze_content_gaps(research_request)
        else:
            raise ValueError(f"Unsupported research type: {research_request.research_type}")

    async def handle_content_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle content generation requests"""
        content_request = ContentRequest(**payload)
        
        self.logger.info(f"Processing content request: {content_request.content_type}")
        
        if content_request.content_type == ContentType.BLOG_POST:
            return await self._generate_blog_post(content_request)
        elif content_request.content_type == ContentType.WEBSITE_COPY:
            return await self._generate_website_copy(content_request)
        elif content_request.content_type == ContentType.SOCIAL_MEDIA:
            return await self._generate_social_media_content(content_request)
        elif content_request.content_type == ContentType.EMAIL_CAMPAIGN:
            return await self._generate_email_campaign(content_request)
        elif content_request.content_type == ContentType.MARKETING_COPY:
            return await self._generate_marketing_copy(content_request)
        else:
            raise ValueError(f"Unsupported content type: {content_request.content_type}")

    async def _perform_keyword_research(self, request: ResearchRequest) -> Dict[str, Any]:
        """Perform comprehensive keyword research"""
        
        # Use the keyword research tool
        keyword_tool = self.tools["keyword_researcher"]
        
        # Primary keyword analysis
        primary_results = await keyword_tool.analyze_keywords(
            seed_keywords=request.keywords,
            target_audience=request.target_audience,
            industry=request.industry_focus,
            geographic_scope=request.geographic_scope
        )
        
        # Competitor keyword analysis
        competitor_keywords = {}
        if request.competitor_urls:
            for url in request.competitor_urls:
                comp_keywords = await keyword_tool.extract_competitor_keywords(url)
                competitor_keywords[url] = comp_keywords
        
        # Long-tail keyword discovery
        long_tail_keywords = await keyword_tool.discover_long_tail_keywords(
            primary_keywords=request.keywords,
            audience_intent=request.content_goals or []
        )
        
        # Keyword difficulty and opportunity analysis
        opportunity_analysis = await keyword_tool.analyze_keyword_opportunities(
            keywords=primary_results.get("keywords", []),
            competitor_data=competitor_keywords
        )
        
        # Generate keyword strategy recommendations
        strategy_prompt = self._build_keyword_strategy_prompt(
            primary_results, competitor_keywords, long_tail_keywords, opportunity_analysis
        )
        
        model = GenerativeModel(self.model_configs["analysis"]["model_name"])
        strategy_response = await model.generate_content_async(
            strategy_prompt,
            generation_config={
                "max_output_tokens": self.model_configs["analysis"]["max_tokens"],
                "temperature": self.model_configs["analysis"]["temperature"]
            }
        )
        
        return {
            "primary_keywords": primary_results,
            "competitor_keywords": competitor_keywords,
            "long_tail_keywords": long_tail_keywords,
            "opportunity_analysis": opportunity_analysis,
            "strategy_recommendations": strategy_response.text,
            "research_metadata": {
                "request_id": str(uuid.uuid4()),
                "timestamp": datetime.utcnow().isoformat(),
                "scope": request.research_type.value
            }
        }

    async def _generate_blog_post(self, request: ContentRequest) -> Dict[str, Any]:
        """Generate SEO-optimized blog post content"""
        
        # Research the topic first
        research_data = await self._research_topic_background(
            topic=request.topic,
            keywords=request.target_keywords,
            audience=request.audience_persona
        )
        
        # Create content outline
        outline_prompt = self._build_blog_outline_prompt(request, research_data)
        
        model = GenerativeModel(self.model_configs["content_generation"]["model_name"])
        outline_response = await model.generate_content_async(
            outline_prompt,
            generation_config={
                "max_output_tokens": 2048,
                "temperature": 0.5
            }
        )
        
        # Generate full content based on outline
        content_prompt = self._build_blog_content_prompt(request, research_data, outline_response.text)
        
        content_response = await model.generate_content_async(
            content_prompt,
            generation_config={
                "max_output_tokens": self.model_configs["content_generation"]["max_tokens"],
                "temperature": self.model_configs["content_generation"]["temperature"]
            }
        )
        
        # SEO optimization
        seo_tool = self.tools["seo_optimizer"]
        optimized_content = await seo_tool.optimize_content(
            content=content_response.text,
            target_keywords=request.target_keywords,
            seo_requirements=request.seo_requirements
        )
        
        # Content validation
        validator = self.tools["content_validator"]
        validation_results = await validator.validate_content(
            content=optimized_content["content"],
            requirements={
                "word_count": request.word_count,
                "tone": request.tone_of_voice,
                "keywords": request.target_keywords
            }
        )
        
        return {
            "outline": outline_response.text,
            "content": optimized_content["content"],
            "seo_analysis": optimized_content["seo_analysis"],
            "validation_results": validation_results,
            "metadata": {
                "word_count": len(optimized_content["content"].split()),
                "readability_score": validation_results.get("readability_score"),
                "seo_score": optimized_content["seo_analysis"].get("overall_score"),
                "generated_at": datetime.utcnow().isoformat()
            }
        }

    def _build_keyword_strategy_prompt(self, primary_results: Dict, competitor_keywords: Dict, 
                                     long_tail_keywords: Dict, opportunity_analysis: Dict) -> str:
        """Build prompt for keyword strategy generation"""
        return f"""
        As an expert SEO strategist, analyze the following keyword research data and provide a comprehensive keyword strategy:

        PRIMARY KEYWORD DATA:
        {json.dumps(primary_results, indent=2)}

        COMPETITOR KEYWORD ANALYSIS:
        {json.dumps(competitor_keywords, indent=2)}

        LONG-TAIL KEYWORD OPPORTUNITIES:
        {json.dumps(long_tail_keywords, indent=2)}

        OPPORTUNITY ANALYSIS:
        {json.dumps(opportunity_analysis, indent=2)}

        Please provide:
        1. Priority keyword clusters for content creation
        2. Content gap opportunities based on competitor analysis
        3. Recommended content calendar focusing on high-opportunity keywords
        4. Technical SEO recommendations for keyword implementation
        5. Measurement and tracking recommendations

        Format your response as structured JSON with clear sections for each recommendation type.
        """

    def _build_blog_outline_prompt(self, request: ContentRequest, research_data: Dict) -> str:
        """Build prompt for blog post outline generation"""
        return f"""
        Create a comprehensive blog post outline for the following requirements:

        TOPIC: {request.topic}
        TARGET KEYWORDS: {', '.join(request.target_keywords)}
        AUDIENCE: {request.audience_persona}
        TONE: {request.tone_of_voice}
        TARGET WORD COUNT: {request.word_count}

        RESEARCH CONTEXT:
        {json.dumps(research_data, indent=2)}

        SEO REQUIREMENTS:
        {json.dumps(request.seo_requirements, indent=2)}

        Create an outline that includes:
        1. Compelling headline with primary keyword
        2. Introduction hook that addresses audience pain points
        3. Main sections with H2/H3 structure incorporating target keywords naturally
        4. Key points to cover in each section
        5. Call-to-action recommendations
        6. Meta description suggestion

        Ensure the outline follows SEO best practices for content structure and keyword distribution.
        """

    async def _research_topic_background(self, topic: str, keywords: List[str], audience: str) -> Dict[str, Any]:
        """Research background information for content creation"""
        
        web_search_tool = self.tools["web_search"]
        trend_analyzer = self.tools["trend_analyzer"]
        
        # Search for current information on the topic
        search_results = await web_search_tool.search_comprehensive(
            query=f"{topic} {' '.join(keywords[:3])}",
            num_results=10,
            include_snippets=True
        )
        
        # Analyze current trends
        trend_data = await trend_analyzer.analyze_topic_trends(
            topic=topic,
            keywords=keywords,
            timeframe="3months"
        )
        
        # Get audience insights
        audience_profiler = self.tools["audience_profiler"]
        audience_insights = await audience_profiler.analyze_audience_interests(
            audience_description=audience,
            topic_context=topic
        )
        
        return {
            "search_results": search_results,
            "trend_data": trend_data,
            "audience_insights": audience_insights,
            "research_timestamp": datetime.utcnow().isoformat()
        }

# Tool Implementations

class WebSearchTool:
    """Advanced web search capabilities for research"""
    
    def __init__(self, project_id: str):
        self.project_id = project_id
        
    async def search_comprehensive(self, query: str, num_results: int = 10, 
                                 include_snippets: bool = True) -> Dict[str, Any]:
        """Perform comprehensive web search"""
        # Implementation would integrate with Google Custom Search API
        # or other search services
        pass

class KeywordResearchTool:
    """Specialized keyword research and analysis"""
    
    def __init__(self, project_id: str):
        self.project_id = project_id
        
    async def analyze_keywords(self, seed_keywords: List[str], target_audience: str,
                             industry: str, geographic_scope: str) -> Dict[str, Any]:
        """Analyze keyword opportunities and metrics"""
        # Implementation would integrate with keyword research APIs
        pass

class ContentGenerationTool:
    """Advanced content generation with multiple model support"""
    
    def __init__(self, project_id: str):
        self.project_id = project_id
        
    async def generate_content_with_model_selection(self, prompt: str, content_type: ContentType,
                                                  requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Generate content with automatic model selection based on requirements"""
        # Implementation would select optimal model based on content type and requirements
        pass

class SEOOptimizationTool:
    """SEO analysis and optimization capabilities"""
    
    def __init__(self, project_id: str):
        self.project_id = project_id
        
    async def optimize_content(self, content: str, target_keywords: List[str],
                             seo_requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize content for SEO performance"""
        # Implementation would analyze and optimize content for SEO factors
        pass
```

## Agent Prompts and Interaction Patterns

### Research Prompts

#### Market Research Prompt Template
```python
MARKET_RESEARCH_PROMPT = """
You are an expert market research analyst specializing in {industry} for {target_audience} in {geographic_region}.

RESEARCH OBJECTIVE:
{research_objective}

CURRENT MARKET CONTEXT:
{market_context}

COMPETITOR LANDSCAPE:
{competitor_data}

Please provide a comprehensive market analysis including:

1. MARKET SIZE AND OPPORTUNITY
   - Total addressable market (TAM) estimation
   - Serviceable addressable market (SAM) analysis
   - Market growth trends and projections
   - Key market drivers and barriers

2. COMPETITIVE LANDSCAPE
   - Direct and indirect competitors analysis
   - Competitive positioning matrix
   - Market share distribution
   - Competitive advantages and weaknesses

3. TARGET AUDIENCE INSIGHTS
   - Detailed buyer personas
   - Pain points and challenges
   - Buying behavior patterns
   - Decision-making criteria

4. MARKET TRENDS AND OPPORTUNITIES
   - Emerging trends affecting the market
   - Technology disruptions
   - Regulatory changes
   - Untapped market segments

5. STRATEGIC RECOMMENDATIONS
   - Market entry strategies
   - Positioning recommendations
   - Pricing strategy insights
   - Go-to-market approach

Format your response as structured JSON with detailed analysis in each section.
Include confidence levels for your assessments and cite data sources where applicable.
"""
```

#### Keyword Research Prompt Template
```python
KEYWORD_RESEARCH_PROMPT = """
You are an expert SEO strategist conducting keyword research for {business_type} targeting {audience_segment}.

BUSINESS CONTEXT:
{business_context}

CURRENT KEYWORD LANDSCAPE:
{existing_keywords}

COMPETITOR KEYWORD DATA:
{competitor_keywords}

RESEARCH PARAMETERS:
- Geographic Focus: {geographic_focus}
- Industry: {industry}
- Content Goals: {content_goals}
- Business Objectives: {business_objectives}

Provide comprehensive keyword research including:

1. PRIMARY KEYWORD OPPORTUNITIES
   - High-volume, relevant keywords
   - Search intent analysis (informational, commercial, transactional)
   - Keyword difficulty assessment
   - Estimated traffic potential

2. LONG-TAIL KEYWORD STRATEGY
   - Specific long-tail variations
   - Question-based keywords
   - Local search opportunities
   - Voice search optimization keywords

3. SEMANTIC KEYWORD CLUSTERS
   - Related terms and synonyms
   - Topic clusters for content planning
   - LSI (Latent Semantic Indexing) keywords
   - Entity-based keyword relationships

4. COMPETITIVE KEYWORD GAPS
   - Keywords competitors rank for that we don't
   - Underutilized keyword opportunities
   - Seasonal keyword trends
   - Emerging keyword opportunities

5. KEYWORD IMPLEMENTATION STRATEGY
   - Priority ranking system
   - Content mapping recommendations
   - Technical implementation guidelines
   - Performance tracking metrics

Present findings in structured JSON format with actionable recommendations for each keyword cluster.
Include search volume estimates, competition levels, and strategic priority scores.
"""
```

### Content Generation Prompts

#### Blog Post Generation Prompt Template
```python
BLOG_POST_PROMPT = """
You are an expert content writer specializing in {industry} content for {target_audience}.

CONTENT BRIEF:
- Topic: {topic}
- Primary Keyword: {primary_keyword}
- Secondary Keywords: {secondary_keywords}
- Target Word Count: {word_count}
- Tone of Voice: {tone}
- Content Goals: {content_goals}

AUDIENCE PROFILE:
{audience_profile}

SEO REQUIREMENTS:
{seo_requirements}

RESEARCH CONTEXT:
{research_data}

BRAND GUIDELINES:
{brand_guidelines}

Create a comprehensive blog post that includes:

1. COMPELLING HEADLINE
   - Include primary keyword naturally
   - Create emotional hook
   - Promise clear value proposition
   - Optimize for click-through rates

2. ENGAGING INTRODUCTION
   - Hook reader with relevant problem/question
   - Establish credibility and expertise
   - Preview article value and structure
   - Include primary keyword in first 100 words

3. STRUCTURED MAIN CONTENT
   - Use clear H2/H3 heading hierarchy
   - Incorporate keywords naturally throughout
   - Include data, statistics, and examples
   - Maintain consistent tone and voice
   - Add internal linking opportunities

4. ACTIONABLE INSIGHTS
   - Provide practical, implementable advice
   - Include step-by-step instructions where relevant
   - Add real-world examples and case studies
   - Address common objections or concerns

5. STRONG CONCLUSION
   - Summarize key takeaways
   - Include clear call-to-action
   - Encourage engagement (comments, shares)
   - Reinforce main value proposition

6. SEO OPTIMIZATION
   - Meta description (150-160 characters)
   - Suggested internal links
   - Image alt text recommendations
   - Schema markup suggestions

Write in {tone} tone, ensuring content is valuable, engaging, and optimized for both search engines and human readers.
Focus on providing genuine value while naturally incorporating target keywords.
"""
```

## Tool Integration Framework

### Google Workspace Integration
```python
class GoogleWorkspaceIntegration:
    """Integration with Google Workspace services"""
    
    def __init__(self, credentials_path: str):
        self.credentials_path = credentials_path
        self.docs_service = self._initialize_docs_service()
        self.sheets_service = self._initialize_sheets_service()
        self.drive_service = self._initialize_drive_service()
    
    async def create_content_document(self, title: str, content: str, 
                                    folder_id: str = None) -> Dict[str, Any]:
        """Create a new Google Doc with generated content"""
        
        # Create document
        document = {
            'title': title
        }
        
        doc = self.docs_service.documents().create(body=document).execute()
        document_id = doc.get('documentId')
        
        # Add content to document
        requests = [
            {
                'insertText': {
                    'location': {
                        'index': 1,
                    },
                    'text': content
                }
            }
        ]
        
        self.docs_service.documents().batchUpdate(
            documentId=document_id,
            body={'requests': requests}
        ).execute()
        
        # Move to specified folder if provided
        if folder_id:
            self.drive_service.files().update(
                fileId=document_id,
                addParents=folder_id,
                removeParents='root'
            ).execute()
        
        return {
            'document_id': document_id,
            'document_url': f'https://docs.google.com/document/d/{document_id}',
            'title': title
        }
    
    async def create_research_spreadsheet(self, title: str, research_data: Dict[str, Any],
                                        folder_id: str = None) -> Dict[str, Any]:
        """Create a Google Sheet with research findings"""
        
        # Create spreadsheet
        spreadsheet = {
            'properties': {
                'title': title
            },
            'sheets': [
                {
                    'properties': {
                        'title': 'Keyword Research'
                    }
                },
                {
                    'properties': {
                        'title': 'Competitor Analysis'
                    }
                },
                {
                    'properties': {
                        'title': 'Content Strategy'
                    }
                }
            ]
        }
        
        sheet = self.sheets_service.spreadsheets().create(body=spreadsheet).execute()
        spreadsheet_id = sheet.get('spreadsheetId')
        
        # Populate with research data
        await self._populate_research_sheets(spreadsheet_id, research_data)
        
        return {
            'spreadsheet_id': spreadsheet_id,
            'spreadsheet_url': f'https://docs.google.com/spreadsheets/d/{spreadsheet_id}',
            'title': title
        }

class VertexAIModelManager:
    """Advanced model management for optimal performance and cost"""
    
    def __init__(self, project_id: str, location: str):
        self.project_id = project_id
        self.location = location
        vertexai.init(project=project_id, location=location)
        
        # Model performance tracking
        self.model_performance = {}
        
    async def select_optimal_model(self, task_type: str, content_length: int,
                                 quality_requirements: Dict[str, Any]) -> str:
        """Select the most appropriate model based on task requirements"""
        
        if task_type == "research_analysis":
            if content_length > 10000:
                return "gemini-1.5-pro"
            else:
                return "gemini-1.5-flash"
                
        elif task_type == "content_generation":
            if quality_requirements.get("creativity", "medium") == "high":
                return "gemini-1.5-pro"
            elif content_length > 5000:
                return "gemini-1.5-pro"
            else:
                return "gemini-1.5-flash"
                
        elif task_type == "data_analysis":
            return "gemini-1.5-flash"
            
        else:
            return "gemini-1.5-pro"  # Default to most capable model
    
    async def generate_with_fallback(self, prompt: str, model_name: str,
                                   generation_config: Dict[str, Any]) -> str:
        """Generate content with automatic fallback to alternative models"""
        
        try:
            model = GenerativeModel(model_name)
            response = await model.generate_content_async(
                prompt,
                generation_config=generation_config
            )
            
            # Track successful generation
            self._track_model_performance(model_name, True, len(response.text))
            
            return response.text
            
        except Exception as e:
            self.logger.warning(f"Model {model_name} failed: {str(e)}")
            
            # Try fallback model
            fallback_model = self._get_fallback_model(model_name)
            if fallback_model:
                try:
                    model = GenerativeModel(fallback_model)
                    response = await model.generate_content_async(
                        prompt,
                        generation_config=generation_config
                    )
                    
                    self._track_model_performance(fallback_model, True, len(response.text))
                    return response.text
                    
                except Exception as fallback_error:
                    self._track_model_performance(fallback_model, False, 0)
                    raise fallback_error
            else:
                self._track_model_performance(model_name, False, 0)
                raise e
```

This implementation provides a comprehensive foundation for the Research & Content Agent, including detailed code blocks, tool integrations, and prompt templates. The agent is designed to work seamlessly within the A2A communication framework while providing powerful research and content generation capabilities.

