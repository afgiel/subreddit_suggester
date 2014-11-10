REPO = $(shell git rev-parse --show-toplevel)

DOWNLOAD_CACHE = ~/.pip_cache
REQS_PROD = requirements/base.txt
REQS_DEV = requirements/dev.txt


clean:
	# Delete all .pyc and .pyo files.
	find ${REPO} \( -name "*~" -o -name "*.py[co]" -o -name ".#*" -o -name "#*#" \) -exec rm '{}' +

pip:
	pip install --upgrade pip==1.5.5
	pip install wheel==0.23.0 --download-cache ${DOWNLOAD_CACHE}

reqs-prod: ${REQS_PROD}
	$(foreach reqs,${REQS_PROD},${BUILD_WHEELS} -r ${reqs}; ${INSTALL_WHEELS} -r ${reqs};)

reqs-dev: ${REQS_DEV}
	$(foreach reqs,${REQS_DEV},${BUILD_WHEELS} -r ${reqs}; ${INSTALL_WHEELS} -r ${reqs};)

reqs: pip reqs-prod reqs-dev