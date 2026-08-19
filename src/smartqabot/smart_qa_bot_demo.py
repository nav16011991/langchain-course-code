from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Demo Usage
from smartqabot.smart_qa_bot import SmartQABot
from langsmith import traceable

def demo_qa_bot():
    bot = SmartQABot()

    questions = [
        "What is the capital of France?",
        "Explain the theory of relativity.",
        "How does photosynthesis work?",
    ]

    print("=" * 60)
    print("SMART Q&A BOT DEMO")
    print("=" * 60)

    for question in questions:

        print(f"\n Question: {question}")
        print("-" * 40)

        response = bot.ask(question)

        print(f"Question: {question}")
        print(f"Answer: {response.answer}")
        print(f"Confidence: {response.confidence}")
        print(f"Reasoning: {response.reasoning}")
        print(f"Follow-up Questions: {response.follow_up_questions}")
        print(f"Sources Needed: {response.sources_needed}")
        print("-" * 60)


@traceable(name="error_handling_demo", run_type="chain")
def demo_error_handling():
    """Demonstrate error handling."""

    bot = SmartQABot()

    print("\n" + "=" * 60)
    print("ERROR HANDLING DEMO")
    print("=" * 60)

    # Test with a very long question (edge case)
    long_question = "What is " + "very " * 100 + "important?"

    response = bot.ask(long_question)
    print(f"Handled gracefully: {response.confidence}")


@traceable(name="batch_demo", run_type="chain")
def demo_batch_processing():
    """Demonstrate batch processing."""

    bot = SmartQABot()

    questions = [
        "What is Python?",
        "What is JavaScript?",
        "What is Rust?",
    ]

    print("\n" + "=" * 60)
    print("BATCH PROCESSING DEMO")
    print("=" * 60)

    responses = bot.ask_batch(questions)

    for q, r in zip(questions, responses):
        print(f"\n{q}")
        print(f"  -> {r.answer[:100]}...")
        print(f"  Confidence: {r.confidence}")

if __name__ == "__main__":

    try:
        demo_qa_bot()
        demo_batch_processing()
        demo_error_handling()

        print("\n" + "=" * 60)
        print("Section 1 Complete!")
        print("=" * 60)
        print(
            """
What you learned:
- LangChain ecosystem overview
- Environment setup with uv
- Core concepts: Runnables, LCEL, pipe operator
- Working with multiple LLM providers
- Prompt templates and message types
- Output parsers and structured output
- Building a production Q&A bot
- LangSmith tracing with @traceable decorator

Next: Section 2 - Chains, RAG & Memory
        """
        )
    finally:
        pass
    # uncomment the line below to flush traces to LangSmith, but you'll alse see an error at the end of a run, which is not harmful at all, but annoying!
    # Client().flush()  # Ensure all traces are sent to LangSmith