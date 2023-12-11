 .PHONY: 

train:
	@python train.py

install:
	@pip install -r requirements.txt

lint:
	@flake8

test_baseline:
	@python test_baseline.py