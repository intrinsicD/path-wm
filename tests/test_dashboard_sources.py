from viewer.dashboard import build_dashboard_artifact
from viewer.ledger import RunResult


def test_large_source_lists_keep_every_exact_path_and_record_identity():
    paths=tuple(f'runs/a_long_experiment_name/source_{i:03d}_with_a_comma,_in_its_filename.json' for i in range(100))
    runs=[RunResult(label='large',kind='source_audit',status='completed',step=None,
                    metrics={},context={},source_paths=paths,modified_at=1.),
          RunResult(label='small',kind='source_audit',status='completed',step=None,
                    metrics={},context={},source_paths=(paths[3],),modified_at=1.)]
    data=build_dashboard_artifact(runs,[])['snapshot']['datasets']
    for inventory in data['inventory']:
        expected=paths if inventory['run']=='large' else (paths[3],)
        actual=tuple(row['path'] for row in data['source_files'] if row['record_key']==inventory['record_key'])
        assert actual==expected
        assert inventory['source_count']==len(expected)
    assert all(len(value)<=4000 for name in ('inventory','source_files') for row in data[name] for value in row.values() if isinstance(value,str))
