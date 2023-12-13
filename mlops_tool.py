import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os
import git
from sklearn.datasets import load_iris
from flask import Flask, render_template, request, redirect, url_for

class MLOpsTool:
    def __init__(self, model_dir='models', data_dir='data', repo_url=None):
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.repo_url = repo_url
        self.repo = None

        if repo_url:
            self.clone_repository()

    def clone_repository(self):
        if os.path.exists('.git'):
            self.repo = git.Repo('.')
        else:
            self.repo = git.Repo.clone_from(self.repo_url, '.')

    def initialize_repository(self):
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)

        if not os.path.exists('.git'):
            repo = git.Repo.init('.')
            repo.index.add(['models', 'data'])
            repo.index.commit('Initial commit with models and data directories')

    def save_model(self, model, version):
        model_path = os.path.join(self.model_dir, f'model_v{version}.joblib')
        joblib.dump(model, model_path)

        if self.repo:
            self.repo.index.add([model_path])
            self.repo.index.commit(f'Add model v{version}')

    def save_dataset(self, dataset, version):
        data_path = os.path.join(self.data_dir, f'data_v{version}.csv')
        dataset.to_csv(data_path, index=False)

        if self.repo:
            self.repo.index.add([data_path])
            self.repo.index.commit(f'Add data v{version}')

    def train_model(self, X, y):
        model = LogisticRegression()
        model.fit(X, y)
        return model

    def evaluate_model(self, model, X, y_true):
        y_pred = model.predict(X)
        accuracy = accuracy_score(y_true, y_pred)
        return accuracy

    def rollback(self, version):
        if self.repo:
            commit_hash = next(self.repo.iter_commits(paths=['models'], max_count=version)).hexsha
            self.repo.git.reset('--hard', commit_hash)
            print(f'Reverted to model version {version}')

    def load_model(self, version):
        model_path = os.path.join(self.model_dir, f'model_v{version}.joblib')
        if os.path.exists(model_path):
            return joblib.load(model_path)
        else:
            print(f'Model version {version} not found.')
            return None

    def load_dataset(self, version):
        data_path = os.path.join(self.data_dir, f'data_v{version}.csv')
        if os.path.exists(data_path):
            return pd.read_csv(data_path)
        else:
            print(f'Dataset version {version} not found.')
            return None






app = Flask(__name__)


mlops_tool = MLOpsTool()
mlops_tool.initialize_repository()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train_and_save', methods=['POST'])
def train_and_save():
    version = int(request.form['version'])

    # Load Iris dataset
    iris = load_iris()
    iris_data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    iris_data['target'] = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        iris_data.drop('target', axis=1), iris_data['target'], test_size=0.2, random_state=42
    )

    # Train and save the model
    model = mlops_tool.train_model(X_train, y_train)
    mlops_tool.save_model(model, version)
    accuracy = mlops_tool.evaluate_model(model, X_test, y_test)

    # Save dataset
    mlops_tool.save_dataset(iris_data, version)

    return redirect(url_for('index'))

@app.route('/rollback', methods=['POST'])
def rollback():
    version = int(request.form['version'])
    mlops_tool.rollback(version)
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
