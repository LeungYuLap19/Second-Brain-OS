from src.agents.orchestrator import OrchestratorAgent

def test_orchestrator():
    orch = OrchestratorAgent()

    test_cases = [
        # =============================================
        # Professor — ONLY when content is PROVIDED
        # =============================================
        ("Summarize the PDF I just uploaded about deep learning.", "Professor"),
        ("Here is a screenshot of my textbook page — explain it.", "Professor"),
        ("Read this text: 'Attention is all you need' paper...", "Professor"),
        ("Attached research paper (PDF) on Grok-4.", "Responder"),
        ("What is a transformer?", "Responder"),  # No text provided
        ("Explain how transformers work.", "Responder"),
        ("Convert this lecture video into notes.", "Responder"),  # Video not supported

        # =============================================
        # Researcher — Pure time-sensitive fact lookup
        # =============================================
        ("Who won the Nobel Prize in Physics 2025?", "Researcher"),
        ("What is the current price of Bitcoin?", "Researcher"),
        ("Compare Gemma2 vs Qwen2.5 performance in December 2025.", "Researcher"),
        ("Latest breakthroughs in fusion energy 2025?", "Researcher"),
        ("What is quantum computing?", "Responder"),  # Known concept, not time-sensitive

        # =============================================
        # Note Taker — ONLY when existing content is referenced
        # =============================================
        ("Here are bullet points from the lecture — turn them into Cornell notes.", "Note Taker"),
        ("Convert this summary into a mind map.", "Note Taker"),
        ("Make a beautiful outline from the previous output above.", "Note Taker"),
        ("Format these raw notes nicely.", "Note Taker"),
        ("Make nice outline notes.", "Responder"),  # No content referenced
        ("Create structured notes from scratch.", "Responder"),

        # =============================================
        # Communicator — ONLY with real email text
        # =============================================
        ("Draft a reply to this email:\nSubject: Re: Meeting tomorrow\nHi team...", "Communicator"),
        ("Summarize this email thread: From: john@... Subject: Project update...", "Communicator"),
        ("Write a professional follow-up email.", "Responder"),  # No email provided
        ("Help me write an email to my boss.", "Responder"),

        # =============================================
        # Concierge — Specific travel only
        # =============================================
        ("Plan a 5-day trip to Tokyo for 2 people in June, budget $4000.", "Concierge"),
        ("Find flights from NYC to London for Christmas week.", "Concierge"),
        ("Best hotels in Bali under $200/night?", "Concierge"),
        ("Recommend a vacation spot.", "Responder"),
        ("I want to travel somewhere warm in December.", "Responder"),  # Too vague

        # =============================================
        # Secretary — Calendar & meeting actions
        # =============================================
        ("Schedule a 1-hour call with Alex next Tuesday at 3pm.", "Secretary"),
        ("Create reminder: Dentist appointment Friday 10am.", "Secretary"),
        ("Here are today's standup notes — extract action items.", "Secretary"),
        ("Move my 2pm sync to tomorrow morning.", "Secretary"),
        ("When is my next meeting?", "Secretary"),

        # =============================================
        # Accountant — Only with real expense data
        # =============================================
        ("Here are 3 receipt images from Tokyo — log expenses.", "Accountant"),
        ("Log expense: $120 hotel, $45 taxi, $80 dinner.", "Accountant"),
        ("I spent $58 on lunch today.", "Accountant"),
        ("Show total for Tokyo trip category.", "Accountant"),
        ("Help me track my expenses.", "Responder"),  # No data

        # =============================================
        # Responder — Catch-all (most common)
        # =============================================
        ("Explain the difference between Llama3 and Gemma2.", "Responder"),
        ("Make this sound more confident: 'I think we could try this...'", "Responder"),
        ("Write a poem about AI.", "Responder"),
        ("Hello! How are you?", "Responder"),
        ("Help me debug this Python code:", "Responder"),
        ("Turn this voice note into text.", "Responder"),
        ("What can you do?", "Responder"),
    ]

    print("\nRunning Orchestrator Routing Test Cases (FINAL STRICT RULES)\n" + "="*80)
    passed = 0
    total = len(test_cases)

    for i, (user_input, expected_agent) in enumerate(test_cases, start=1):
        output = orch.run(user_input)
        next_agent = output.get("next_agent") if isinstance(output, dict) else None
        status = "PASS" if next_agent == expected_agent else "FAIL"
        if status == "PASS":
            passed += 1
        print(f"{i:2d}. {status} | Expected: {expected_agent.ljust(12)} | Got: {str(next_agent).ljust(12)} → {user_input[:75]}{'...' if len(user_input)>75 else ''}")

    print("="*80)
    print(f"FINAL RESULT: {passed}/{total} TESTS PASSED", end="")
    if passed == total:
        print(" — PERFECT ROUTING ACHIEVED!")
    else:
        print()
    print("="*80)


if __name__ == "__main__":
    test_orchestrator()