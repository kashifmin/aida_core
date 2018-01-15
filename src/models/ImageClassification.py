class ImageClassificationModel:
    def __init__(self):
        self.model = self.__loadModel()

    def __loadModel(self):
        pass

    def classify(imageData):
        # predictions = self.model.predict()

        predictions = [{
            'description': 'car',
            'score': 0.8114 },{
            'description': 'bike',
            'score': 0.9114 }]
        return predictions