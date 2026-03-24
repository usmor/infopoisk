async function switchLanguage(lang) {
  try {
    localStorage.setItem("selectedLanguage", lang);

    const response = await fetch(`/api/translate?lang=${lang}`);
    const translations = await response.json();

    document.querySelectorAll("[data-i18n]").forEach((el) => {
      const key = el.getAttribute("data-i18n");

      let value = translations;
      const keys = key.split(".");
      for (const k of keys) {
        if (value && typeof value === "object") {
          value = value[k];
        } else {
          value = null;
          break;
        }
      }

      if (value) {
        el.textContent = value;
      }
    });

    document.documentElement.lang = lang;
  } catch (error) {
    console.error("Translation error:", error);
    window.location.href = `?lang=${lang}`;
  }
}

document.addEventListener("DOMContentLoaded", () => {
  document
    .querySelectorAll('input[name="language-switch"]')
    .forEach((radio) => {
      radio.addEventListener("change", (e) => {
        if (e.target.checked) {
          switchLanguage(e.target.value);
        }
      });
    });

  const savedLang = localStorage.getItem("selectedLanguage");
  if (savedLang) {
    const radio = document.querySelector(
      `input[name="language-switch"][value="${savedLang}"]`,
    );
    if (radio && !radio.checked) {
      radio.checked = true;
      switchLanguage(savedLang);
    }
  }
});
