document.addEventListener('DOMContentLoaded', () => {
    const summarizeBtn = document.getElementById('summarize-btn');
    const codeInput = document.getElementById('code-input');
    const summaryText = document.getElementById('summary-text');
    const resultArea = document.getElementById('result-area');
    const btnLoader = document.getElementById('btn-loader');
    const btnText = summarizeBtn.querySelector('.btn-text');

    summarizeBtn.addEventListener('click', async () => {
        const code = codeInput.value.trim();

        if (!code) {
            alert('Please paste some code first!');
            return;
        }

        // UI Feedback: Loading state
        summarizeBtn.disabled = true;
        btnLoader.style.display = 'block';
        btnText.style.opacity = '0.5';

        try {
            const response = await fetch('/api/summarize', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ code }),
            });

            const data = await response.json();

            if (data.error) {
                summaryText.textContent = `Error: ${data.error}`;
                summaryText.style.color = '#f87171';
            } else {
                summaryText.textContent = data.summary;
                summaryText.style.color = '#e2e8f0';
            }

            // Show result with animation
            resultArea.classList.add('show');

        } catch (error) {
            console.error('Error:', error);
            summaryText.textContent = 'Failed to connect to the backend server.';
            summaryText.style.color = '#f87171';
            resultArea.classList.add('show');
        } finally {
            // UI Feedback: Reset state
            summarizeBtn.disabled = false;
            btnLoader.style.display = 'none';
            btnText.style.opacity = '1';
        }
    });

    // Handle CMD+Enter or CTRL+Enter to summarize
    codeInput.addEventListener('keydown', (e) => {
        if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
            summarizeBtn.click();
        }
    });
});
