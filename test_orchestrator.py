from src.agents.orchestrator import OrchestratorAgent

def test_orchestrator():
  orch = OrchestratorAgent()

  # Example test cases: (user_input, list of expected agents in plan)
  test_cases = [
    # Professor
    ("Summarize the PDF I just uploaded about transformers.", ["Professor"]),
    ("Here is a paper on AI: please summarize.", ["Professor"]),
    ("Analyze this Word document about quantum computing.", ["Professor"]),
    
    # Responder - concept explanation / vague
    ("What is a transformer?", ["Responder"]),
    ("Explain blockchain in simple terms.", ["Responder"]),
    ("Write a short poem about AI.", ["Responder"]),
    
    # Note Taker
    ("Turn these bullet points into Cornell notes.", ["Note Taker"]),
    ("Make a mind map from the previous summary.", ["Note Taker"]),
    ("Convert these notes into a structured outline.", ["Note Taker"]),
    
    # Communicator
    ("Draft a reply to this email: Subject: Meeting tomorrow...", ["Communicator"]),
    ("Summarize this email thread from HR.", ["Communicator"]),
    ("Reply to John: see the attached message.", ["Communicator"]),
    
    # Concierge - concrete travel
    ("Plan a trip to Bali for 2 people in December, budget $3000.", ["Concierge"]),
    ("Find best hotels in Paris under $200 for June.", ["Concierge"]),
    ("Book flights to New York from July 10 to 15.", ["Concierge"]),
    
    # Responder - vague travel
    ("I want to travel somewhere nice.", ["Responder"]),
    ("Suggest a place to go on vacation.", ["Responder"]),
    ("I need a warm holiday destination.", ["Responder"]),
    
    # Secretary
    ("Schedule a call with Sarah next Wednesday 3pm.", ["Secretary"]),
    ("Create reminder: Dentist Friday 10am.", ["Secretary"]),
    ("Move my 2pm sync to tomorrow.", ["Secretary"]),
    
    # Accountant
    ("Log expense: $58 lunch, $220 hotel.", ["Accountant"]),
    ("Show total spending on Conference 2025.", ["Accountant"]),
    ("Create new category: Daily Expenses.", ["Accountant"]),
    
    # Composite requests (multi-agent)
    ("Summarize the PDF and then turn it into point form notes.", ["Professor", "Note Taker"]),
    ("Read this paper, summarize it, and explain key points to me.", ["Professor", "Responder"]),
    ("Plan a trip to Bali, then draft an itinerary email for me.", ["Concierge", "Communicator"]),
    ("Summarize my emails and schedule meetings based on them.", ["Communicator", "Secretary"]),
    
    # Edge / tricky
    ("Explain transformers and then write a poem about them.", ["Responder"]),
    ("I have receipts and notes, summarize notes and log expenses.", ["Note Taker", "Accountant"]),
    ("Turn this PDF into a mind map and schedule a meeting about it.", ["Professor", "Note Taker", "Secretary"]),
  ]


  print("\nRunning Orchestrator Routing Test Cases\n" + "="*80)
  passed = 0
  total = len(test_cases)

  for i, (user_input, expected_agents) in enumerate(test_cases, start=1):
      try:
        output = orch.run(user_input)
        tasks = output.get("tasks", [])
        agents_in_plan = [t.get("agent") for t in tasks if "agent" in t]

        status = "PASS" if agents_in_plan == expected_agents else "FAIL"
        if status == "PASS":
          passed += 1
        else:
          print(f"{i:2d}. {status} | Expected: {expected_agents} | Got: {agents_in_plan} → {user_input[:75]}{'...' if len(user_input) > 75 else ''}")

          # Print instructions for each task
          for t in tasks:
            agent = t.get("agent")
            instr = t.get("instruction")
            inputs = t.get("inputs", [])
            output_key = t.get("output")
            print(f"    -> Agent: {agent}, Inputs: {inputs}, Output: {output_key}")
            print(f"       Instruction: {instr}")

      except Exception as e:
        print(f"{i:2d}. ERROR | {user_input[:75]}... | Exception: {e}")

  print("="*80)
  print(f"FINAL RESULT: {passed}/{total} TESTS PASSED", end="")
  if passed == total:
    print(" — PERFECT ROUTING ACHIEVED!")
  else:
    print()
  print("="*80)


if __name__ == "__main__":
  test_orchestrator()
