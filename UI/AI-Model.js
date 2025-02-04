const fileInput = document.getElementById("fileInput");
        const submitBtn = document.getElementById("submitBtn");
        const statusText = document.getElementById("status");
        const resultsTable = document.getElementById("resultsTable");
        const resultsBody = document.getElementById("resultsBody");

        submitBtn.addEventListener("click", async () => {
            const file = fileInput.files[0];
            if (!file) {
                alert("Please select a file.");
                return;
            }

            console.log("📂 Selected file:", file.name);

            const formData = new FormData();
            formData.append("file", file);

            statusText.textContent = "Uploading...";
            resultsTable.style.display = "none"; // Hide previous results

            try {
                const response = await fetch("http://127.0.0.1:5000/predict", {
                    method: "POST",
                    body: formData
                });

                const data = await response.json();
                console.log("🔍 Server Response:", data);

                if (data.error) {
                    statusText.textContent = "Error: " + data.error;
                    return;
                }

                // Show results
                resultsBody.innerHTML = data.prediction.map((pred, i) => ` 
                    <tr>
                        <td>Sample ${i + 1}</td>
                        <td>${pred}</td>
                        <td>${(data.confidence[i] * 100).toFixed(2)}%</td>
                    </tr>
                `).join("");

                resultsTable.style.display = "block";
                statusText.textContent = "✅ Predictions Ready!";
            } catch (error) {
                console.error("❌ Fetch error:", error);
                statusText.textContent = "Server error!";
            }
        });