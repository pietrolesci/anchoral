sources = src scripts energizer notebooks

format:
	ruff format $(sources)

lint:
	ruff check $(sources) --fix

clean-poetry-cache:
	rm -rf ~/.cache/pypoetry/virtualenvs/