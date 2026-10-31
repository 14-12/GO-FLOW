"""Traffic-light link maps (cosim/tlmap.py)."""

from cosim.tlmap import links_from_landmark_ids, links_from_net


def test_landmark_ids_split_on_the_last_underscore_of_a_known_tls():
    ids = ["cluster_1_2_3", "cluster_1_2_10", "gneJ5_0", "nope_x", "unknown_4"]
    links = links_from_landmark_ids(ids, ["cluster_1_2", "gneJ5"])
    assert links == {"cluster_1_2_3": [("cluster_1_2", 3)], "cluster_1_2_10": [("cluster_1_2", 10)],
                     "gneJ5_0": [("gneJ5", 0)]}


def test_link_signal_params(tmp_path):
    net = tmp_path / "n.net.xml"
    net.write_text("""<net>
      <tlLogic id="77" type="static" programID="0" offset="0">
        <phase duration="42" state="gr"/>
        <param key="linkSignalID:0" value="1001"/>
        <param key="linkSignalID:1" value="1002"/>
      </tlLogic>
      <tlLogic id="77" type="static" programID="webster" offset="0">
        <phase duration="30" state="rg"/>
        <param key="linkSignalID:0" value="1001"/>
      </tlLogic>
      <tlLogic id="a:b" type="static" programID="0" offset="0">
        <param key="linkSignalID:0" value="1001"/>
        <param key="other" value="x"/>
      </tlLogic>
    </net>""")
    assert links_from_net(net) == {"1001": [("77", 0), ("a:b", 0)], "1002": [("77", 1)]}
