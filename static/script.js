document.getElementById("generate-form").addEventListener("submit", function (event) {
    event.preventDefault();

    const formData = new FormData(event.target);
    const resultDiv = document.getElementById("result");
    resultDiv.innerHTML = "Generating images...";

    fetch("/generate", {
        method: "POST",
        body: formData,
    })
        .then((response) => response.json())
        .then((data) => {
            resultDiv.innerHTML = "";
            if (data.error) {
                resultDiv.innerHTML = `<p>Error: ${data.error}</p>`;
                return;
            }

            data.images.forEach((image) => {
                const img = document.createElement("img");
                img.src = image;
                resultDiv.appendChild(img);
            });
        })
        .catch((error) => {
            console.error("Error:", error);
            resultDiv.innerHTML = "<p>Something went wrong. Please try again.</p>";
        });
});
