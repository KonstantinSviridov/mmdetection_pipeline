from musket_core import datasets,image_datasets

@datasets.dataset_provider(origin="train.csv",kind="MultiClassInstanceSegmentationDataSet")
def getTrain():
    return image_datasets.InstanceSegmentationDataSet(["train"],"train.csv","ImageId","EncodedPixels","ClassId", classes=["0", "1", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "2", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "3", "30", "31", "32", "33", "34", "35", "36", "37", "38", "39", "4", "40", "41", "42", "43", "44", "45", "5", "6", "7", "8", "9"])