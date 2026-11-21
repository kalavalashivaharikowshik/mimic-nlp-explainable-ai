/**
 * ICU Mortality Prediction - frontend logic.
 *
 * All server data is rendered with textContent / DOM nodes (never innerHTML),
 * so text from the note or the API can never inject HTML or scripts.
 */
(() => {
    "use strict";

    // Relative URLs work on localhost, 127.0.0.1, Docker and any hosting domain.
    const API = { predict: "predict", health: "health" };

    const FEATURES = [
        { id: "f0", name: "age", label: "Age", unit: "years" },
        { id: "f1", name: "gender", label: "Gender", unit: "" },
        { id: "f2", name: "heart_rate", label: "Heart rate", unit: "bpm" },
        { id: "f3", name: "sbp", label: "Systolic BP", unit: "mmHg" },
        { id: "f4", name: "dbp", label: "Diastolic BP", unit: "mmHg" },
        { id: "f5", name: "mbp", label: "Mean BP", unit: "mmHg" },
        { id: "f6", name: "spo2", label: "SpO2", unit: "%" },
        { id: "f7", name: "creatinine", label: "Creatinine", unit: "mg/dL" },
        { id: "f8", name: "glucose", label: "Glucose", unit: "mg/dL" },
        { id: "f9", name: "hemoglobin", label: "Hemoglobin", unit: "g/dL" },
    ];
    const UNITS = Object.fromEntries(FEATURES.map((f) => [f.name, f.unit]));

    const SAMPLES = {
        stable: {
            text: "Patient is stable, vitals within normal range, no signs of infection. Tolerating diet, ambulating independently. Plan for transfer to the floor tomorrow.",
            features: [45, 0, 76, 122, 78, 93, 98, 0.9, 104, 14.1],
        },
        critical: {
            text: "78 y/o admitted with septic shock from pneumonia. Hypotensive despite fluids, started on norepinephrine. Intubated overnight for hypoxic respiratory failure. Acute kidney injury with rising creatinine, oliguric.",
            features: [78, 1, 122, 84, 46, 58, 87, 3.1, 248, 8.2],
        },
    };

    const $ = (id) => document.getElementById(id);
    const form = $("patientForm");
    const textInput = $("text");
    const resultDiv = $("result");
    const predictBtn = $("predictBtn");
    const formErrors = $("formErrors");

    /** Tiny DOM helper: el("p", {class: "x"}, "text", childNode, ...) */
    function el(tag, attrs = {}, ...children) {
        const node = document.createElement(tag);
        for (const [key, value] of Object.entries(attrs)) {
            if (value === undefined || value === null) continue;
            if (key === "class") node.className = value;
            else if (key === "style") node.setAttribute("style", value);
            else node.setAttribute(key, value);
        }
        for (const child of children.flat()) {
            if (child === null || child === undefined || child === false) continue;
            node.append(child instanceof Node ? child : document.createTextNode(String(child)));
        }
        return node;
    }

    // ------------------------------------------------------------------
    // Model status
    // ------------------------------------------------------------------
    let modelReady = false;

    function setStatus(kind, text) {
        const pill = $("modelStatus");
        pill.className = `status-pill status-${kind}`;
        pill.querySelector(".status-text").textContent = text;
    }

    async function checkHealth(attempt = 0) {
        try {
            const res = await fetch(API.health, { cache: "no-store" });
            const data = await res.json();
            if (data.model === "ready") {
                modelReady = true;
                setStatus("ready", "Model ready");
                return;
            }
            if (data.model === "error") {
                setStatus("error", "Model failed to load");
                return;
            }
            setStatus("loading", "Loading model… (first start can take a minute)");
        } catch {
            setStatus("error", "Server unreachable");
        }
        // Poll while loading; back off gently.
        setTimeout(() => checkHealth(attempt + 1), Math.min(3000 + attempt * 1000, 10000));
    }

    // ------------------------------------------------------------------
    // Form helpers
    // ------------------------------------------------------------------
    function fillSample(name) {
        const sample = SAMPLES[name];
        textInput.value = sample.text;
        FEATURES.forEach((f, i) => { $(f.id).value = String(sample.features[i]); });
        updateCharCount();
        clearErrors();
    }

    function clearForm() {
        form.reset();
        updateCharCount();
        clearErrors();
        resultDiv.replaceChildren();
    }

    function updateCharCount() {
        $("charCount").textContent = textInput.value.length.toLocaleString();
    }

    function clearErrors() {
        formErrors.hidden = true;
        formErrors.replaceChildren();
        document.querySelectorAll(".input-group.invalid").forEach((g) => g.classList.remove("invalid"));
    }

    function showErrors(messages) {
        formErrors.replaceChildren(
            el("strong", {}, "Please fix the following:"),
            el("ul", {}, messages.map((m) => el("li", {}, m))),
        );
        formErrors.hidden = false;
        formErrors.scrollIntoView({ behavior: "smooth", block: "center" });
    }

    function collectInput() {
        clearErrors();
        const errors = [];
        const features = [];
        let firstInvalid = null;

        for (const f of FEATURES) {
            const input = $(f.id);
            const raw = input.value.trim();
            const value = Number(raw);
            let problem = null;

            if (raw === "" || Number.isNaN(value)) {
                problem = `${f.label} is required`;
            } else if (input.min !== "" && value < Number(input.min)) {
                problem = `${f.label} must be at least ${input.min} ${f.unit}`.trim();
            } else if (input.max !== "" && value > Number(input.max)) {
                problem = `${f.label} must be at most ${input.max} ${f.unit}`.trim();
            }

            if (problem) {
                errors.push(problem);
                input.closest(".input-group").classList.add("invalid");
                firstInvalid = firstInvalid || input;
            }
            features.push(value);
        }

        if (!errors.length && features[4] > features[3]) {
            errors.push("Diastolic BP cannot be higher than systolic BP");
            $("f4").closest(".input-group").classList.add("invalid");
            firstInvalid = $("f4");
        }

        if (errors.length) {
            showErrors(errors);
            firstInvalid.focus({ preventScroll: true });
            return null;
        }
        return { text: textInput.value.trim(), features };
    }

    // ------------------------------------------------------------------
    // Rendering
    // ------------------------------------------------------------------
    function fmtValue(name, value) {
        if (name === "gender") return value === 1 ? "female" : "male";
        const digits = name === "creatinine" || name === "hemoglobin" ? 1 : 0;
        const unit = UNITS[name] || "";
        return unit === "%" ? `${value.toFixed(digits)}%` : `${value.toFixed(digits)} ${unit}`.trim();
    }

    function riskCard(data) {
        const high = data.prediction === "HIGH RISK";
        const pct = (data.probability * 100).toFixed(1);
        const thr = (data.threshold * 100).toFixed(0);
        const pos = Math.min(Math.max(data.probability * 100, 0), 100);

        return el("div", { class: `risk-card ${high ? "high" : "low"}` },
            el("p", { class: "eyebrow" }, "Mortality risk assessment"),
            el("h2", {}, data.prediction),
            el("p", { class: "risk-prob" }, "Predicted mortality probability: ", el("strong", {}, `${pct}%`)),
            el("div", { class: "meter", role: "img", "aria-label": `Probability ${pct} percent, threshold ${thr} percent` },
                el("span", { class: "meter-threshold", style: `left:${data.threshold * 100}%` }),
                el("span", { class: "meter-marker", style: `left:${pos}%` }),
            ),
            el("div", { class: "meter-legend" }, el("span", {}, "0%"), el("span", {}, `threshold ${thr}%`), el("span", {}, "100%")),
            el("p", { class: "risk-explain" },
                high
                    ? `Classified HIGH RISK because the probability is at or above the ${thr}% decision threshold.`
                    : `Classified LOW RISK because the probability is below the ${thr}% decision threshold.`),
            data.borderline
                ? el("p", { class: "borderline", role: "note" },
                    "⚠ Borderline: the probability is close to the threshold, so small changes in the inputs could change the label. Interpret with extra caution.")
                : null,
        );
    }

    function highlightedNote(text, words) {
        const box = el("div", { class: "note-preview" });
        if (!text) {
            box.append(el("em", {}, "No clinical note provided."));
            return box;
        }
        if (!words.length) {
            box.textContent = text;
            return box;
        }
        const escaped = words.map((w) => w.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
        const pattern = new RegExp(`\\b(${escaped.join("|")})\\b`, "gi");
        let last = 0;
        for (const match of text.matchAll(pattern)) {
            box.append(document.createTextNode(text.slice(last, match.index)));
            box.append(el("mark", {}, match[0]));
            last = match.index + match[0].length;
        }
        box.append(document.createTextNode(text.slice(last)));
        return box;
    }

    function keyTermsBox(data, noteText) {
        const terms = data.explanation.key_terms || [];
        return el("div", { class: "result-box" },
            el("h3", {}, "Key terms in the note"),
            el("p", { class: "box-caption" }, "Words the text model attended to most (attention is a hint, not proof of cause)."),
            terms.length
                ? el("div", { class: "terms" }, terms.map((t) =>
                    el("span", { class: "term", style: `--alpha:${(0.08 + 0.22 * t.score).toFixed(2)}`, title: `relative attention ${t.score.toFixed(2)}` }, t.word)))
                : el("p", { class: "empty" }, "No informative terms found."),
            highlightedNote(noteText, terms.map((t) => t.word)),
        );
    }

    function driversBox(data) {
        const items = data.explanation.top_features || [];
        return el("div", { class: "result-box" },
            el("h3", {}, "Top vital & lab drivers"),
            el("p", { class: "box-caption" }, "Largest SHAP contributions for this patient (percentage points of risk)."),
            el("ul", { class: "driver-list" }, items.map((f) => {
                const pp = f.contribution * 100;
                const cls = f.direction === "increases risk" ? "up" : f.direction === "decreases risk" ? "down" : "none";
                const arrow = cls === "up" ? "▲" : cls === "down" ? "▼" : "•";
                return el("li", { class: "driver" },
                    el("span", {}, el("span", { class: "driver-name" }, f.label), " ", el("span", { class: "driver-value" }, fmtValue(f.name, f.value))),
                    el("span", { class: `driver-effect ${cls}`, title: f.direction }, `${arrow} ${pp >= 0 ? "+" : ""}${pp.toFixed(2)} pp`),
                );
            })),
        );
    }

    function flagsBox(data) {
        const flags = data.clinical_flags || [];
        return el("div", { class: "result-box" },
            el("h3", {}, "Clinical flags"),
            el("p", { class: "box-caption" }, "Transparent rule-based checks using standard thresholds. These are not model output."),
            flags.length
                ? el("ul", { class: "flag-list" }, flags.map((f) =>
                    el("li", { class: "flag" }, el("span", { class: "flag-source" }, f.source), el("span", {}, f.message))))
                : el("p", { class: "empty" }, "No abnormal values or high-risk terms detected."),
        );
    }

    /**
     * Diverging horizontal bar chart drawn with plain HTML (accessible, crisp
     * on every screen). items: [{label, value}], value in probability units.
     */
    function chartBox(title, caption, items, footnote) {
        const max = Math.max(...items.map((d) => Math.abs(d.value)), 1e-9);
        const rows = items.map((d) => {
            const pct = (Math.abs(d.value) / max) * 50;
            const cls = d.value > 1e-6 ? "up" : d.value < -1e-6 ? "down" : "none";
            const pp = d.value * 100;
            const text = `${pp >= 0 ? "+" : ""}${pp.toFixed(2)} pp`;
            return el("div", { class: "chart-row", role: "listitem", "aria-label": `${d.label}: ${text}` },
                el("span", { class: "chart-label", title: d.label }, d.label),
                el("span", { class: "chart-track", "aria-hidden": "true" },
                    el("span", { class: `chart-bar ${cls}`, style: cls === "down" ? `right:50%;width:${pct}%` : `left:50%;width:${pct}%` }),
                ),
                el("span", { class: `chart-value ${cls}` }, text),
            );
        });
        return el("div", { class: "result-box" },
            el("h3", {}, title),
            el("p", { class: "box-caption" }, caption),
            el("div", { class: "chart", role: "list" }, rows),
            el("div", { class: "chart-legend", "aria-hidden": "true" },
                el("span", {}, el("i", { class: "swatch down" }), "decreases risk"),
                el("span", {}, el("i", { class: "swatch up" }), "increases risk"),
            ),
            footnote ? el("p", { class: "box-caption", style: "margin:10px 0 0" }, footnote) : null,
        );
    }

    function shapItems(data, raw) {
        const names = FEATURES.map((f) => f.name);
        return names
            .map((name, i) => ({
                label: `${FEATURES[i].label} = ${fmtValue(name, raw[i])}`,
                value: data.shap.values[name] ?? 0,
            }))
            .sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
    }

    function limeItems(data) {
        return Object.entries(data.lime.weights)
            .map(([label, value]) => ({ label, value }))
            .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
            .slice(0, 6);
    }

    function renderResult(data, noteText, raw) {
        const wrapper = el("div", { class: "fade-in" },
            riskCard(data),
            el("div", { class: "result-grid" }, keyTermsBox(data, noteText), driversBox(data)),
            flagsBox(data),
            chartBox(
                "SHAP explanation (this patient)",
                `Exact contribution of each vital/lab value, starting from the average reference patient (${(data.shap.base_value * 100).toFixed(1)}%) and adding up to this patient's ${(data.probability * 100).toFixed(1)}%.`,
                shapItems(data, raw),
            ),
            chartBox(
                "LIME explanation (local rules)",
                "Simple rules that approximate the model's behaviour around this patient.",
                limeItems(data),
                `Local fit R² = ${data.lime.fit_r2.toFixed(2)} (closer to 1 = the rules describe the model better).`,
            ),
            el("p", { class: "footnote" }, data.note_message),
            el("p", { class: "footnote" }, data.disclaimer),
        );
        resultDiv.replaceChildren(wrapper);
        resultDiv.scrollIntoView({ behavior: "smooth", block: "start" });
    }

    function renderError(title, messages) {
        resultDiv.replaceChildren(
            el("div", { class: "risk-card error fade-in" },
                el("h3", {}, title),
                el("ul", { style: "margin:10px 0 0 18px;font-size:14px" }, messages.map((m) => el("li", {}, m))),
            ),
        );
    }

    function renderLoading() {
        resultDiv.replaceChildren(
            el("div", { class: "loading-box" },
                el("div", { class: "loader", "aria-hidden": "true" }),
                el("p", {}, "Running NLP analysis and computing SHAP / LIME explanations…"),
            ),
        );
    }

    // ------------------------------------------------------------------
    // Submit
    // ------------------------------------------------------------------
    async function onSubmit(event) {
        event.preventDefault();
        const payload = collectInput();
        if (!payload) return;

        predictBtn.disabled = true;
        const originalText = predictBtn.textContent;
        predictBtn.textContent = "Analyzing patient data…";
        resultDiv.setAttribute("aria-busy", "true");
        renderLoading();

        try {
            const response = await fetch(API.predict, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });

            let data = null;
            try { data = await response.json(); } catch { /* non-JSON error page */ }

            if (!response.ok) {
                const detail = data && data.detail;
                const messages = Array.isArray(detail) ? detail : [detail || `Server returned ${response.status}`];
                if (response.status === 422) {
                    resultDiv.replaceChildren();
                    showErrors(messages);
                } else if (response.status === 503) {
                    renderError("Model not ready", messages);
                    checkHealth();
                } else {
                    renderError("Analysis failed", messages);
                }
                return;
            }

            renderResult(data, payload.text, payload.features);
            if (!modelReady) { modelReady = true; setStatus("ready", "Model ready"); }
        } catch (err) {
            console.error("Prediction error:", err);
            renderError("Could not reach the server", [
                "Check your connection, or make sure the backend is running (uvicorn backend.main:app).",
            ]);
        } finally {
            predictBtn.disabled = false;
            predictBtn.textContent = originalText;
            resultDiv.setAttribute("aria-busy", "false");
        }
    }

    // ------------------------------------------------------------------
    // Wire up
    // ------------------------------------------------------------------
    form.addEventListener("submit", onSubmit);
    textInput.addEventListener("input", updateCharCount);
    document.querySelectorAll("[data-sample]").forEach((btn) =>
        btn.addEventListener("click", () => fillSample(btn.dataset.sample)));
    $("clearBtn").addEventListener("click", clearForm);
    $("calcMbp").addEventListener("click", () => {
        const sbp = Number($("f3").value);
        const dbp = Number($("f4").value);
        if ($("f3").value && $("f4").value) $("f5").value = String(Math.round((sbp + 2 * dbp) / 3));
        else showErrors(["Enter systolic and diastolic BP first to calculate mean BP"]);
    });
    form.querySelectorAll("input, select").forEach((input) =>
        input.addEventListener("input", () => input.closest(".input-group")?.classList.remove("invalid")));

    updateCharCount();
    checkHealth();
})();
