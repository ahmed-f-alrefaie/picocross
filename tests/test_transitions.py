import pytest

@pytest.fixture
def transfile(tmp_path_factory):
    fn = tmp_path_factory.mktemp('data') / 'test.states'
    fn.write_text(
"""       43046        40053 3.2870e-06
       29200        27275 3.1410e-06
      113689       101839 1.2720e-05
       30020        27326 1.7260e-03
       53901        58033 4.3970e-07
       55919        56129 6.6740e-09
"""


    )

    return fn



def test_load_exomol_transition(transfile):
    from picocross.transitions import load_exomol_transition

    df = load_exomol_transition(transfile)

    assert len(df.columns) == 3
    assert 'upper' in df.columns
    assert 'lower' in df.columns
    assert 'Afi' in df.columns
    assert len(df) == 6


    assert df['upper'][0] == 43046
    assert df['lower'][3] == 27326
    assert df['Afi'][4] == 4.3970e-07


def test_load_exomol_transition_chunks(transfile):
    from picocross.transitions import load_exomol_transition

    df_chunk = load_exomol_transition(transfile, chunksize=2)

    for df in df_chunk:
        assert len(df.columns) == 3
        assert 'upper' in df.columns
        assert 'lower' in df.columns
        assert 'Afi' in df.columns
        assert len(df) == 2