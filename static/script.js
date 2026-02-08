document.getElementById("predictionForm").addEventListener("submit", async function (e) {
    e.preventDefault();

    const inputs = document.querySelectorAll("#predictionForm input");

    const data = {
        age: Number(inputs[0].value),
        gender: Number(inputs[1].value),
        bmi: Number(inputs[2].value),
        sbp: Number(inputs[3].value),
        dbp: Number(inputs[4].value),
        fpg: Number(inputs[5].value),
        ffpg: Number(inputs[6].value),
        chol: Number(inputs[7].value),
        tri: Number(inputs[8].value),
        hdl: Number(inputs[9].value),
        ldl: Number(inputs[10].value),
        smoking: Number(inputs[11].value),
        drinking: Number(inputs[12].value),
        family_history: Number(inputs[13].value)
    };

    const response = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data)
    });

    const result = await response.json();

    document.getElementById("result").innerHTML = `
        <p><b>ML Probability of Diabetes:</b> ${result.probability}</p>
        <p><b>Final Diagnosis:</b> ${result.diagnosis}</p>
    `;
});
