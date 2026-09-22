document.addEventListener('DOMContentLoaded', () => {
  // Theme toggling logic
  const themeBtn = document.getElementById('theme-btn');
  const htmlEl = document.documentElement;
  
  // Check local storage for theme
  if (localStorage.getItem('theme') === 'light') {
    htmlEl.classList.remove('dark');
  }

  themeBtn.addEventListener('click', () => {
    htmlEl.classList.toggle('dark');
    if (htmlEl.classList.contains('dark')) {
      localStorage.setItem('theme', 'dark');
    } else {
      localStorage.setItem('theme', 'light');
    }
  });

  // Form submission logic
  const form = document.getElementById('prediction-form');
  const submitBtn = document.getElementById('submit-btn');
  const spinner = document.getElementById('spinner');
  const btnText = submitBtn.querySelector('span');
  
  const resultPanel = document.getElementById('result-panel');
  const resultTitle = document.getElementById('result-title');
  const resultDesc = document.getElementById('result-desc');
  const probBar = document.getElementById('prob-bar');

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    // UI state loading
    btnText.style.display = 'none';
    spinner.style.display = 'block';
    submitBtn.disabled = true;
    resultPanel.classList.add('hidden');
    probBar.style.width = '0%';

    // Gather form data
    const formData = new FormData(form);
    const data = Object.fromEntries(formData.entries());
    
    try {
      const response = await fetch('/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
      });
      
      const result = await response.json();
      
      if (result.success) {
        // Show result
        resultPanel.style.display = 'block';
        // brief delay for display block to register before removing hidden class for transition
        setTimeout(() => {
          resultPanel.classList.remove('hidden');
        }, 50);
        
        const prob = result.probability * 100;
        
        if (result.prediction === 1) {
          resultTitle.textContent = "High Interest";
          resultTitle.style.color = "var(--accent)";
          resultDesc.textContent = `The model predicts this customer is highly likely to be interested in vehicle insurance (${prob.toFixed(1)}% probability).`;
          probBar.style.backgroundColor = "var(--accent)";
        } else {
          resultTitle.textContent = "Low Interest";
          resultTitle.style.color = "var(--muted)";
          resultDesc.textContent = `The model predicts this customer is unlikely to be interested in vehicle insurance (${prob.toFixed(1)}% probability).`;
          probBar.style.backgroundColor = "var(--muted)";
        }
        
        // Animate progress bar
        setTimeout(() => {
          probBar.style.width = `${prob}%`;
        }, 300);
        
      } else {
        alert("Error making prediction: " + result.error);
      }
    } catch (error) {
      console.error(error);
      alert("Failed to connect to the server.");
    } finally {
      // Reset UI state
      btnText.style.display = 'block';
      spinner.style.display = 'none';
      submitBtn.disabled = false;
    }
  });
});
