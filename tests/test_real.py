import copy
import json
from pathlib import Path
import pytest
from perception_replay.real import iou, detection_metrics, evaluation_predictions, summarize, validate_annotations, frame_at


def test_matching_is_one_to_one_and_counts_duplicates_as_false_positives():
    box = [0,0,20,40]
    predictions = [{'box':box,'score':.9},{'box':box,'score':.8}]
    assert detection_metrics([box],predictions)=={'tp':1,'fp':1,'fn':0}
    assert iou(box,box)==1
    assert iou(box,[30,0,40,40])==0


def config():
    return {'evaluation_roi':{'x':0,'y':0,'width':100,'height':100},'min_height':10}


def test_scoring_filters_do_not_mutate_detector_outputs():
    predictions = [{'box':[0,0,20,40],'score':.9},{'box':[150,0,170,40],'score':.99},{'box':[30,0,50,40],'score':.1}]
    before = copy.deepcopy(predictions)
    annotation={'boxes':[[0,0,20,40]],'ignored_regions':[]}
    assert len(evaluation_predictions(predictions,annotation,config(),.5))==1
    assert predictions==before


def test_ignored_region_never_removes_a_valid_truth_match():
    annotation={'boxes':[[0,0,20,40]],'ignored_regions':[[0,0,21,41]]}
    assert len(evaluation_predictions([{'box':[0,0,20,40],'score':.9}],annotation,config(),.5))==1


def test_development_summary_does_not_use_held_out_predictions():
    annotations={0:{'split':'development','boxes':[[0,0,20,40]],'ignored_regions':[]},10000:{'split':'test','boxes':[[0,0,20,40]],'ignored_regions':[]}}
    rows=[{'time_ms':0,'models':{'model':{'detections':[{'box':[0,0,20,40],'score':.9}]}}},{'time_ms':10000,'models':{'model':{'detections':[]}}}]
    first=summarize(rows,annotations,config(),'model',.5,'development')
    rows[1]['models']['model']['detections']=[{'box':[0,0,20,40],'score':1}]
    assert summarize(rows,annotations,config(),'model',.5,'development')==first
    assert first['recall']==1


def test_authored_annotations_have_unique_split_times_and_valid_boxes():
    path=Path(__file__).resolve().parents[1]/'data/annotations.json'
    annotations=json.loads(path.read_text(encoding='utf-8'))
    validate_annotations(annotations)
    dev=[frame['time_ms'] for frame in annotations['frames'] if frame['split']=='development']
    test=[frame['time_ms'] for frame in annotations['frames'] if frame['split']=='test']
    assert max(dev)<min(test)
    assert len(dev)==len(test)==4
    annotations['frames'].append(copy.deepcopy(annotations['frames'][0]))
    with pytest.raises(ValueError): validate_annotations(annotations)


def test_negative_frame_time_is_rejected_before_decoding():
    with pytest.raises(ValueError): frame_at(Path('missing.mp4'),-1)
