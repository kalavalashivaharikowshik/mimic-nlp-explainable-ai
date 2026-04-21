/**
 * MIMIC NLP Prediction Engine
 * Professional Healthcare UI Integration
 */

async function predict(event) {
    // Prevent default form behavior if necessary
    if (event) event.preventDefault();

    const predictBtn = document.getElementById("predictBtn");
    const resultDiv = document.getElementById("result");
    const textInput = document.getElementById("text");

    // 1. INPUT VALIDATION & DATA GATHERING
    const text = textInput.value.trim();
    if (!text) {
        alert("Please enter clinical observation notes.");
        textInput.focus();
        return;
    }

    let features = [];
    for (let i = 0; i < 10; i++) {
        const inputElement = document.getElementById("f" + i);
        const val = parseFloat(inputElement.value);

        if (isNaN(val)) {
            // Find the label text to tell the user exactly what is missing
            const label = inputElement.previousElementSibling;
            const fieldName = label ? label.innerText : `Feature ${i+1}`;
            
            alert(`Please enter a valid number for: ${fieldName}`);
            inputElement.focus();
            return;
        }
        features.push(val);
    }

    // 2. UI STATE: LOADING
    // Disable button to prevent double-submissions
    predictBtn.disabled = true;
    const originalBtnText = predictBtn.innerText;
    predictBtn.innerText = "Analyzing Patient Data...";
    
    // Inject modern loader
    resultDiv.innerHTML = `
        <div class="loader"></div>
        <p style="text-align:center; color: #64748b; font-weight: 500; margin-top: 15px;">
            Running NLP analysis and processing physiological vitals...
        </p>
    `;

    try {
        // 3. API REQUEST
        const response = await fetch("http://127.0.0.1:8000/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text, features })
        });

        if (!response.ok) throw new Error("Backend API connection failed.");

        const data = await response.json();

        // 4. DYNAMIC UI RENDERING
        const riskClass = data.prediction === "HIGH RISK" ? "high" : "low";

        resultDiv.innerHTML = `
            <div class="risk-card ${riskClass}">
                <p style="text-transform: uppercase; letter-spacing: 1.5px; font-weight: 700; font-size: 11px; margin-bottom: 8px; opacity: 0.8;">
                    Mortality Risk Prediction
                </p>
                <h2 style="margin:0">${data.prediction}</h2>
                <p style="margin-top: 10px; font-size: 18px;">
                    <strong>Confidence Score:</strong> ${(data.probability * 100).toFixed(2)}%
                </p>
            </div>

            <div class="result-grid">
                <div class="result-box">
                    <h4>Clinical NLP Insights</h4>
                    <p style="color: var(--brand-primary); font-weight: 600; font-size: 15px; line-height: 1.6;">
                        ${data.explanation.important_words.join(" • ")}
                    </p>
                </div>
                <div class="result-box">
                    <h4>Top Physiological Drivers</h4>
                    <p style="color: var(--brand-secondary); font-weight: 600; font-size: 15px; line-height: 1.6;">
                        ${data.explanation.top_features.join(" • ")}
                    </p>
                </div>
            </div>

            <div class="result-box" style="margin-top:25px;">
                <h4>SHAP Interpretation (Global Impact)</h4>
                <img src="data:image/png;base64,${data.shap_plot}" alt="SHAP Plot" />
            </div>

            <div class="result-box" style="margin-top:25px;">
                <h4>LIME Explanation (Local Interpretation)</h4>
                <img src="data:image/png;base64,${data.lime_plot}" alt="LIME Plot" />
            </div>
        `;

        // 5. SMOOTH SCROLL TO RESULTS
        // Ensures the user sees the output immediately on mobile/small screens
        resultDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

    } catch (err) {
        console.error("Prediction Error:", err);
        resultDiv.innerHTML = `
            <div class="risk-card" style="background: #fef2f2; color: #991b1b; border: 1px solid #fee2e2;">
                <h3 style="margin:0">Analysis Interrupted</h3>
                <p style="margin-top:10px; font-size: 14px;">
                    Could not connect to the analysis server. Please ensure your Python backend is running at <strong>localhost:8000</strong>.
                </p>
            </div>
        `;
    } finally {
        // 6. RESTORE BUTTON STATE
        predictBtn.disabled = false;
        predictBtn.innerText = originalBtnText;
    }
}

// Attach the event listener to the prediction button
document.getElementById("predictBtn").addEventListener("click", predict);