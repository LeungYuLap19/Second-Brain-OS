```
second brain OS
├─ configs
│  ├─ ollama_models.yaml
│  ├─ settings_loader.py
│  ├─ system_prompts
│  │  ├─ accountant_prompt.txt
│  │  ├─ communicator_prompt.txt
│  │  ├─ orchestrator_prompt.txt
│  │  ├─ professor_prompt.txt
│  │  ├─ researcher_prompt.txt
│  │  ├─ responder_prompt.txt
│  │  ├─ secretary_prompt.txt
│  │  └─ synthesizer_prompt.txt
│  └─ __pycache__
│     └─ settings_loader.cpython-313.pyc
├─ src
│  ├─ agents
│  │  ├─ accountant.py
│  │  ├─ base_agent.py
│  │  ├─ communicator.py
│  │  ├─ orchestrator.py
│  │  ├─ professor.py
│  │  ├─ researcher.py
│  │  ├─ responder.py
│  │  ├─ secretary.py
│  │  ├─ synthesizer.py
│  │  ├─ __init__.py
│  │  └─ __pycache__
│  │     ├─ base_agent.cpython-313.pyc
│  │     ├─ communicator.cpython-313.pyc
│  │     ├─ note_taker.cpython-313.pyc
│  │     ├─ orchestrator.cpython-313.pyc
│  │     ├─ plan_reviewer.cpython-313.pyc
│  │     ├─ professor.cpython-313.pyc
│  │     ├─ researcher.cpython-313.pyc
│  │     ├─ responder.cpython-313.pyc
│  │     ├─ synthesizer.cpython-313.pyc
│  │     └─ __init__.cpython-313.pyc
│  ├─ main.py
│  ├─ managers
│  │  ├─ workflow_manager.py
│  │  ├─ __init.py
│  │  └─ __pycache__
│  │     └─ workflow_manager.cpython-313.pyc
│  ├─ schemas
│  │  ├─ task_state.py
│  │  ├─ __init__.py
│  │  └─ __pycache__
│  │     ├─ task_state.cpython-313.pyc
│  │     └─ __init__.cpython-313.pyc
│  ├─ tools
│  │  ├─ doc_tools.py
│  │  ├─ gmail.py
│  │  ├─ registry.py
│  │  ├─ tavily.py
│  │  ├─ __init__.py
│  │  └─ __pycache__
│  │     ├─ doc_ingest.cpython-313.pyc
│  │     ├─ doc_tools.cpython-313.pyc
│  │     ├─ gmail.cpython-313.pyc
│  │     ├─ registry.cpython-313.pyc
│  │     ├─ tavily.cpython-313.pyc
│  │     └─ __init__.cpython-313.pyc
│  ├─ utils
│  │  ├─ helper.py
│  │  ├─ __init__.py
│  │  └─ __pycache__
│  │     ├─ helper.cpython-313.pyc
│  │     └─ __init__.cpython-313.pyc
│  ├─ __init__.py
│  └─ __pycache__
│     └─ __init__.cpython-313.pyc
├─ test_workflow.py