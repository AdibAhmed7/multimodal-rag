# Contributing to Multimodal RAG

Thank you for your interest in contributing to the Multimodal RAG project! We welcome contributions, feature suggestions, bug reports, and optimizations.

---

## Code of Conduct

Please maintain a respectful, welcoming, and inclusive environment when interacting with the community and maintainers.

---

## Development Workflow

1. **Fork and Clone**
   ```bash
   git clone https://github.com/<your-username>/multimodal-rag.git
   cd multimodal-rag
   ```

2. **Environment Setup**
   ```bash
   python -m venv .venv
   # Windows:
   .\.venv\Scripts\Activate.ps1
   # Linux/macOS:
   source .venv/bin/activate

   pip install -r requirements.txt
   pip install pytest pytest-cov black flake8
   ```

3. **Branching Strategy**
   - Create a feature branch: `git checkout -b feature/<feature-name>`
   - Create a bugfix branch: `git checkout -b fix/<bug-name>`

4. **Code Quality and Tests**
   - Ensure code is formatted cleanly (PEP 8 / Black).
   - Add unit tests for any new features in `tests/unit/`.
   - Run tests before committing:
     ```bash
     pytest -v
     ```

5. **Commit Conventions**
   Follow Conventional Commits:
   - `feat: add hybrid dense-sparse retriever`
   - `fix: resolve bounding box coordinate misalignment`
   - `docs: update quick start instructions`
   - `test: add unit tests for image chunker`

6. **Submit a Pull Request**
   - Push your branch to GitHub.
   - Open a Pull Request detailing the changes, reasoning, and test results.
