load_docs_scripts:
	if [ ! -d "docs-scripts" ] ; then \
		git clone -b scripts https://github.com/Nixtla/docs.git docs-scripts --single-branch; \
	fi

api_docs:
	lazydocs .neuralforecast --no-watermark
	python docs/to_mdx.py

examples_docs:
	mkdir -p nbs/_extensions
	cp -r docs-scripts/mintlify/ nbs/_extensions/mintlify
	python docs-scripts/update-quarto.py
	quarto render nbs/docs --output-dir ../docs/mintlify/

format_docs:
	# replace _docs with docs
	sed -i -e 's/_docs/docs/g' ./docs-scripts/docs-final-formatting.bash
	bash ./docs-scripts/docs-final-formatting.bash
	find docs/mintlify -name "*.mdx" -exec sed -i.bak '/^:::/d' {} + && find docs/mintlify -name "*.bak" -delete

preview_docs:
	cd docs/mintlify && mintlify dev

all_docs: load_docs_scripts api_docs examples_docs format_docs
