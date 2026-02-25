# How the Code Works (Simplified Walkthrough)

## 1. Input: The Images
*   **What happens:** The code (`src/data_generator.py`) creates 5,000 small pictures (64x64 pixels).
*   **The Logic:**
    *   If it draws a **fuzzy blob**, it labels it `0` (Elliptical).
    *   If it draws **swirly lines**, it labels it `1` (Spiral).
*   **Analogy:** It's like a teacher drawing 5,000 shapes on a chalkboard and writing "Circle" or "Spiral" under each one.

## 2. The Learning: The Model
*   **What happens:** The code (`src/model.py`) defines a **Convolutional Neural Network (CNN)**.
*   **The Logic:**
    *   The model looks at the first 4,000 images (Training Set).
    *   It makes a guess.
    *   It checks the answer key (the label).
    *   If it's wrong, it adjusts its internal numbers (weights) slightly.
    *   It repeats this thousands of times until it stops making mistakes.
*   **Analogy:** A student looking at flashcards. At first, they guess randomly. Over time, they realize "Oh, whenever I see swirly lines, the answer is Spiral."

## 3. The Test: Validation
*   **What happens:** The code (`src/main.py`) takes the remaining 1,000 images (Validation Set) that the model has **never seen before**.
*   **The Logic:**
    *   It shows the image to the model.
    *   The model says "I think this is a Spiral."
    *   The code checks if it was actually a Spiral.
    *   It calculates the **Accuracy** (how many it got right).
*   **Analogy:** The final exam. The teacher gives new questions to see if the student actually learned the concept or just memorized the answers.

## 4. The Output: Predictions
*   **What happens:** The code saves the trained model (`galaxy_classifier.pth`) and produces a plot (`prediction_samples.png`).
*   **The Value:** Now you have a "machine" that can take **any** new image of a galaxy and tell you what it is, instantly.
