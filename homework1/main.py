from search import search

# Точка входа приложения (запуск функции поиска)
if __name__ == "__main__":
    # Запуск функции поиска с ограничением длины snippetа до 1000 символов
    search(snippet_len=1000)
    # search()  # Если не нужно ограничивать длину snippetа
