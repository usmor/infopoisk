document.addEventListener("DOMContentLoaded", function () {
  async function loadTranslation(talkId, lang, contentDiv) {
    try {
      const response = await fetch(
        `/api/talk_translation?talk_id=${talkId}&lang=${lang}`,
      );
      const data = await response.json();

      if (data.error) {
        throw new Error(data.error);
      }

      contentDiv.innerHTML = "";

      const descriptionLabel = data.labels?.description_label || "Описание";
      const transcriptLabel = data.labels?.transcript_label || "Транскрипт";

      const titleElement = document.createElement("h3");
      titleElement.className = "strong";
      titleElement.textContent = data.title;
      contentDiv.appendChild(titleElement);

      if (data.description) {
        const descriptionContainer = document.createElement("p");

        const descriptionStrong = document.createElement("strong");
        descriptionStrong.className = "strong";
        descriptionStrong.textContent = `📝 ${descriptionLabel}: `;

        const descriptionText = document.createTextNode(data.description);

        descriptionContainer.appendChild(descriptionStrong);
        descriptionContainer.appendChild(descriptionText);

        contentDiv.appendChild(descriptionContainer);
      }

      if (data.transcript) {
        const transcriptContainer = document.createElement("p");

        const transcriptStrong = document.createElement("strong");
        transcriptStrong.className = "strong";
        transcriptStrong.textContent = `📄 ${transcriptLabel}: `;

        const transcriptText = document.createTextNode(data.transcript);

        transcriptContainer.appendChild(transcriptStrong);
        transcriptContainer.appendChild(transcriptText);

        contentDiv.appendChild(transcriptContainer);
      }
    } catch (error) {
      console.error("Ошибка:", error);
      contentDiv.innerHTML = `<p>❌ Ошибка загрузки: ${error.message}</p>`;
    }
  }
  const langDetails = document.querySelectorAll(".language-details");
  console.log(`Найдено details: ${langDetails.length}`);

  langDetails.forEach((details) => {
    details.addEventListener("toggle", function () {
      if (this.open) {
        const talkId = this.getAttribute("data-talk-id");
        const lang = this.getAttribute("data-lang");
        const contentDiv = this.querySelector(".language-content");

        if (contentDiv.innerHTML.includes("Загрузка...")) {
          loadTranslation(talkId, lang, contentDiv);
        }
      }
    });
  });
});
