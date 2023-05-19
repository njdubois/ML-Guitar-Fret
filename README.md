training and test images must be 150x150 px in size.

images with frets and strings must be named `fret_{num-frets-in-image}_strings_{num-strings-in-image}_{image-index}.png`

images that do not have frets/strings need to be named in the following format `not_fret_{image-index}.png`

run `python3 trainer.py` to run the trainer

run `python3 tester.py` to run the tester.