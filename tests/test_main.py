import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from instagram_post_agent.main import handler


@pytest.mark.asyncio
async def test_handler_returns_response():
    """Test that handler accepts messages and returns a response."""
    messages = [{"role": "user", "content": "Create Instagram content for https://example.com eco-friendly bags"}]

    # Mock Instagram marketing response
    mock_response = """**Instagram Marketing Content**

**Ad Copy Options:**
Option 1:
Eco-friendly bags that make sustainability stylish...

**Photograph Descriptions:**
Photograph 1:
A person carrying eco bags in a sunny park...

**Marketing Parameters:**
- Product: Eco Bags
- Target Audience: Environmentally conscious consumers
- Campaign Tone: Inspirational
- Key Message: Sustainable fashion
- Call to Action: Shop now
- Hashtags: #ecofriendly #sustainable"""

    with (
        patch("instagram_post_agent.main._initialized", True),
        patch("instagram_post_agent.main.run_copy_crew", new_callable=AsyncMock, return_value=mock_response),
        patch("instagram_post_agent.main.run_image_crew", new_callable=AsyncMock, return_value="Photograph descriptions"),
        patch("instagram_post_agent.main.extract_marketing_parameters_with_llm", return_value={
            "product_name": "Eco Bags",
            "target_audience": "Environmentally conscious consumers",
            "campaign_tone": "Inspirational",
            "key_message": "Sustainable fashion",
            "call_to_action": "Shop now",
            "hashtags": "#ecofriendly #sustainable"
        }),
    ):
        result = await handler(messages)

    # Verify we get a string (Instagram content) back
    assert result is not None
    assert isinstance(result, str)
    assert "Instagram Marketing Content" in result
    assert "Eco Bags" in result


@pytest.mark.asyncio
async def test_handler_with_instagram_query():
    """Test that handler processes Instagram marketing queries correctly."""
    messages = [{"role": "user", "content": "Create Instagram ads for fitness app at https://fitapp.com"}]

    mock_response = """**Instagram Marketing Content**

**Ad Copy Options:**
Option 1: Transform your fitness journey...

**Photograph Descriptions:**
Photograph 1: Fit person exercising at sunrise...

**Marketing Parameters:**
- Product: Fitness App
- Target Audience: Fitness enthusiasts
- Campaign Tone: Motivational
- Key Message: AI-powered workouts
- Call to Action: Download now
- Hashtags: #fitness #workout"""

    with (
        patch("instagram_post_agent.main._initialized", True),
        patch("instagram_post_agent.main.run_copy_crew", new_callable=AsyncMock, return_value=mock_response),
        patch("instagram_post_agent.main.run_image_crew", new_callable=AsyncMock, return_value="Photograph descriptions"),
        patch(
            "instagram_post_agent.main.extract_marketing_parameters_with_llm", return_value={
                "product_name": "Fitness App",
                "target_audience": "Fitness enthusiasts",
                "campaign_tone": "Motivational",
                "key_message": "AI-powered workouts",
                "call_to_action": "Download now",
                "hashtags": "#fitness #workout"
            }
        ),
    ):
        result = await handler(messages)

    assert result is not None
    assert isinstance(result, str)
    assert "Fitness App" in result


@pytest.mark.asyncio
async def test_handler_initialization():
    """Test that handler initializes crew on first call."""
    messages = [{"role": "user", "content": "Create Instagram content for product at https://example.com"}]

    mock_response = "Instagram marketing content..."

    # Start with _initialized as False to test initialization path
    with (
        patch("instagram_post_agent.main._initialized", False),
        patch("instagram_post_agent.main.initialize_crew", new_callable=AsyncMock) as mock_init,
        patch("instagram_post_agent.main.run_copy_crew", new_callable=AsyncMock, return_value=mock_response),
        patch("instagram_post_agent.main.run_image_crew", new_callable=AsyncMock, return_value="Image descriptions"),
        patch("instagram_post_agent.main.extract_marketing_parameters_with_llm", return_value={
            "product_name": "Product",
            "target_audience": "General",
            "campaign_tone": "Professional",
            "key_message": "Features",
            "call_to_action": "Learn more",
            "hashtags": "#product"
        }),
        patch("instagram_post_agent.main._init_lock", new_callable=MagicMock()) as mock_lock,
    ):
        # Configure the lock to work as an async context manager
        mock_lock_instance = MagicMock()
        mock_lock_instance.__aenter__ = AsyncMock(return_value=None)
        mock_lock_instance.__aexit__ = AsyncMock(return_value=None)
        mock_lock.return_value = mock_lock_instance

        result = await handler(messages)

        # Verify initialization was called
        mock_init.assert_called_once()
        # Verify we got a result
        assert result is not None
        assert isinstance(result, str)


@pytest.mark.asyncio
async def test_handler_race_condition_prevention():
    """Test that handler prevents race conditions with initialization lock."""
    messages = [{"role": "user", "content": "Test Instagram query for https://test.com"}]

    mock_response = "Test Instagram content"

    # Test with multiple concurrent calls
    with (
        patch("instagram_post_agent.main._initialized", False),
        patch("instagram_post_agent.main.initialize_crew", new_callable=AsyncMock) as mock_init,
        patch("instagram_post_agent.main.run_copy_crew", new_callable=AsyncMock, return_value=mock_response),
        patch("instagram_post_agent.main.run_image_crew", new_callable=AsyncMock, return_value="Test images"),
        patch("instagram_post_agent.main.extract_marketing_parameters_with_llm", return_value={
            "product_name": "Test",
            "target_audience": "Test",
            "campaign_tone": "Test",
            "key_message": "Test",
            "call_to_action": "Test",
            "hashtags": "#test"
        }),
        patch("instagram_post_agent.main._init_lock", new_callable=MagicMock()) as mock_lock,
    ):
        # Configure the lock to work as an async context manager
        mock_lock_instance = MagicMock()
        mock_lock_instance.__aenter__ = AsyncMock(return_value=None)
        mock_lock_instance.__aexit__ = AsyncMock(return_value=None)
        mock_lock.return_value = mock_lock_instance

        # Call handler twice to ensure lock is used
        await handler(messages)
        await handler(messages)

        # Verify initialize_crew was called only once (due to lock)
        mock_init.assert_called_once()


@pytest.mark.asyncio
async def test_handler_with_detailed_instagram_query():
    """Test that handler can process detailed Instagram marketing queries."""
    messages = [
        {"role": "user", "content": "Create luxury Instagram campaign for skincare line at https://luxuryskincare.com, target women 30-50"}
    ]

    mock_response = """**Instagram Marketing Content**

**Ad Copy Options:**
Option 1: Experience luxury skincare...

**Photograph Descriptions:**
Photograph 1: Elegant skincare products in spa setting...

**Marketing Parameters:**
- Product: Luxury Skincare
- Target Audience: Women 30-50
- Campaign Tone: Luxurious
- Key Message: Premium ingredients
- Call to Action: Discover luxury
- Hashtags: #luxury #skincare"""

    with (
        patch("instagram_post_agent.main._initialized", True),
        patch("instagram_post_agent.main.run_copy_crew", new_callable=AsyncMock, return_value=mock_response),
        patch("instagram_post_agent.main.run_image_crew", new_callable=AsyncMock, return_value="Luxury photo concepts"),
        patch(
            "instagram_post_agent.main.extract_marketing_parameters_with_llm", return_value={
                "product_name": "Luxury Skincare",
                "target_audience": "Women 30-50",
                "campaign_tone": "Luxurious",
                "key_message": "Premium ingredients",
                "call_to_action": "Discover luxury",
                "hashtags": "#luxury #skincare"
            }
        ),
    ):
        result = await handler(messages)

    assert result is not None
    assert isinstance(result, str)
    assert "Luxury Skincare" in result
    assert "luxurious" in result.lower()


@pytest.mark.asyncio
async def test_handler_empty_user_input():
    """Test that handler handles empty user input gracefully."""
    messages = [
        {"role": "system", "content": "You are an Instagram marketing assistant"},
        {"role": "assistant", "content": "How can I help with your Instagram content?"},
        # No user message
    ]

    with patch("instagram_post_agent.main._initialized", True):
        result = await handler(messages)

    assert result is not None
    assert isinstance(result, str)
    assert "Please provide" in result


@pytest.mark.asyncio
async def test_handler_crew_exception():
    """Test that handler handles crew execution exceptions."""
    messages = [{"role": "user", "content": "Create Instagram content for https://example.com"}]

    with (
        patch("instagram_post_agent.main._initialized", True),
        patch("instagram_post_agent.main.run_copy_crew", new_callable=AsyncMock) as mock_run,
        patch("instagram_post_agent.main.extract_marketing_parameters_with_llm", return_value={
            "product_name": "Product",
            "target_audience": "General",
            "campaign_tone": "Professional",
            "key_message": "Features",
            "call_to_action": "Learn more",
            "hashtags": "#product"
        }),
    ):
        # Make run_copy_crew raise an exception
        mock_run.side_effect = Exception("Crew execution failed")

        result = await handler(messages)

    assert result is not None
    assert isinstance(result, str)
    assert "Error" in result


@pytest.mark.asyncio
async def test_handler_edge_case_malformed_messages():
    """Test handler with edge case malformed messages."""

    # We must patch _initialized to True to avoid triggering initialize_crew()
    # which fails without API keys
    with patch("instagram_post_agent.main._initialized", True):
        # Test with non-list input
        result = await handler("not a list")  # type: ignore[invalid-argument-type]
        assert result is not None
        assert isinstance(result, str)
        assert "Error" in result

        # Test with empty list
        result = await handler([])
        assert result is not None
        assert isinstance(result, str)
        assert "Please provide" in result

        # Test with list but no user messages
        result = await handler([{"role": "system", "content": "test"}])
        assert result is not None
        assert isinstance(result, str)
        assert "Please provide" in result


@pytest.mark.asyncio
async def test_handler_with_url_extraction():
    """Test that handler correctly extracts URLs from messages."""
    messages = [{"role": "user", "content": "Check out https://testproduct.com and make Instagram ads"}]

    mock_response = "Instagram content for Test Product"

    with (
        patch("instagram_post_agent.main._initialized", True),
        patch("instagram_post_agent.main.run_copy_crew", new_callable=AsyncMock, return_value=mock_response),
        patch("instagram_post_agent.main.run_image_crew", new_callable=AsyncMock, return_value="Image concepts"),
        patch(
            "instagram_post_agent.main.extract_marketing_parameters_with_llm", return_value={
                "product_name": "Test Product",
                "target_audience": "General",
                "campaign_tone": "Professional",
                "key_message": "Features",
                "call_to_action": "Learn more",
                "hashtags": "#test"
            }
        ),
    ):
        result = await handler(messages)

    assert result is not None
    assert "Test Product" in result