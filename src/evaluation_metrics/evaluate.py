from evaluation_metrics.detA import multi_video_detA
from evaluation_metrics.hota import multi_video_hota

def evaluate(output_dir, dataset_dir):
    """
    Evaluate DetA, HOTA, and AssA metrics across multiple videos in the dataset.
    """

    detA_results = multi_video_detA(output_dir, dataset_dir)
    hota_assa_results = multi_video_hota(output_dir, dataset_dir)

    result = {}
    result.update(detA_results)
    result.update(hota_assa_results)
    return result