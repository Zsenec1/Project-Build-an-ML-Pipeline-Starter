import pytest
import wandb
import pandas as pd
import os

def pytest_addoption(parser):
    parser.addoption("--csv", action="store", default="clean_sample.csv:latest")
    parser.addoption("--ref", action="store", default="clean_sample.csv:reference")
    parser.addoption("--kl_threshold", action="store", default=0.2, type=float)
    parser.addoption("--min_price", action="store", default=10, type=float)
    parser.addoption("--max_price", action="store", default=350, type=float)

def pytest_configure(config):
    config.option.csv = config.getoption("--csv")
    config.option.ref = config.getoption("--ref")
    config.option.kl_threshold = config.getoption("--kl_threshold")
    config.option.min_price = config.getoption("--min_price")
    config.option.max_price = config.getoption("--max_price")

@pytest.fixture(scope='session')
def data(request):
    run = wandb.init(job_type="data_tests", resume=True)
    
    # Download input artifact safely for Windows (avoiding colon path error)
    artifact = run.use_artifact(request.config.option.csv)
    download_dir = "./wandb_download_check"
    artifact.download(root=download_dir)
    
    entry_name = list(artifact.manifest.entries.keys())[0]
    data_path = os.path.join(download_dir, entry_name)
    
    df = pd.read_csv(data_path)
    return df

@pytest.fixture(scope='session')
def ref_data(request):  # <-- Renamed from ref to ref_data
    run = wandb.init(job_type="data_tests", resume=True)
    
    # Download reference artifact safely for Windows
    artifact = run.use_artifact(request.config.option.ref)
    download_dir = "./wandb_download_ref"
    artifact.download(root=download_dir)
    
    entry_name = list(artifact.manifest.entries.keys())[0]
    ref_path = os.path.join(download_dir, entry_name)
    
    df = pd.read_csv(ref_path)
    return df

@pytest.fixture(scope='session')
def kl_threshold(request):
    return request.config.option.kl_threshold

@pytest.fixture(scope='session')
def min_price(request):
    return request.config.option.min_price

@pytest.fixture(scope='session')
def max_price(request):
    return request.config.option.max_price