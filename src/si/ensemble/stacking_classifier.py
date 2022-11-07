from si.data.dataset import Dataset

class StackingClassifier:

    def __init__(self, models, final_model):
        self.models = models
        self.final_model = final_model


    def fit(self, dataset_train:Dataset, dataset_test:Dataset) -> 'StackingClassifier':

        pred_vals = []
        for m in self.models:
            m.fit(dataset_train)
            pred_vals.append(m.predict(dataset_test))