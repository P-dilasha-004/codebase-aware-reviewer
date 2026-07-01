# Agent Instructions and Development Guidelines

## 1. Planning & Architecture Validation (planning.md)
* **Roadmap Adherence:** You must strictly follow the specifications and steps outlined in `planning.md` as the primary project roadmap.
* **Robustness Check:** Continuously evaluate the approaches dictated in `planning.md`. If any architectural decision, implementation step, or logic seems non-robust, brittle, or sub-optimal, you must **halt execution**.
* **Direct Challenge:** Ask the human-in-the-loop for clarification before proceeding. When doing so, provide the most honest, unbiased feedback possible. Do not sugarcoat your critique of the planned approach; clearly explain *why* it is not robust and present a better alternative to drive continuous improvement.

## 2. Strict Sequential Execution & Human-in-the-Loop Verification
* **Step-by-Step Verification:** You must **never** proceed to the next step of coding or implementation before fully verifying the initial or current steps. 
* **Mandatory Approval:** All major architectural decisions, feature completions, and final outputs must be verified by the human-in-the-loop (the user) before moving forward.
* **Iterative Review:** Pause at logical milestones to request human review, ensuring alignment with project goals.

## 3. Edge Case Discussion and Handling
* **Proactive Analysis:** Before implementing features, brainstorm and outline potential edge cases, boundary conditions, and failure modes.
* **Collaborative Mitigation:** Discuss these edge cases explicitly with the human-in-the-loop to agree on expected behavior and fallback mechanisms.
* **Robustness:** Implement safeguards against unexpected inputs and ensure the system fails gracefully rather than crashing catastrophically.

## 4. Test-Driven Development (TDD) for Backend
* **Test-First Philosophy:** Write unit and integration tests for backend components *before* writing the actual implementation code.
* **Comprehensive Coverage:** Ensure tests cover both happy paths and the previously discussed edge cases.
* **Validation:** Code should only be considered complete when all pre-written tests pass successfully.

## 5. Code Quality Standards
* **Readable and Clean:** Write modular, well-structured code. Use clear, descriptive naming conventions for variables, functions, and classes. Prioritize readability over clever but obscure one-liners.
* **Well-Commented:** Include meaningful comments that explain the *why* behind complex logic, algorithms, or architectural choices. Avoid redundant comments that merely restate the code.
* **Maintainability:** Adhere to language-specific style guides and best practices to ensure the codebase remains maintainable over time.