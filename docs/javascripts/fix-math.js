// Fix RST-style math blocks from NumPy docstrings
document.addEventListener('DOMContentLoaded', function() {
  // Find all code blocks that might contain math
  const codeBlocks = document.querySelectorAll('pre code');

  codeBlocks.forEach(block => {
    const text = block.textContent;

    // Check if it looks like a math block (contains LaTeX-like syntax)
    if (text.includes('\\') && (text.includes('\\mathbf') || text.includes('\\sigma') || text.includes('\\frac'))) {
      // Create a new div with math class
      const mathDiv = document.createElement('div');
      mathDiv.className = 'arithmatex';
      mathDiv.innerHTML = '\\[' + text.trim() + '\\]';

      // Replace the code block with the math div
      block.parentElement.replaceWith(mathDiv);
    }
  });

  // Retrigger MathJax typesetting
  if (window.MathJax) {
    MathJax.typesetPromise();
  }
});
