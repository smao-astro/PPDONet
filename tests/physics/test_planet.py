import onet_disk2D.physics.planet


class TestReadPlanetConfig:
    def test1(self):
        pcfg = onet_disk2D.physics.planet.read_planet_config("jupiter.cfg", "Jupiter")
        assert pcfg["Mass"] == 0.001

    def test2(self):
        pcfg = onet_disk2D.physics.planet.read_planet_config(
            "jupiter.cfg",
        )
        assert pcfg["Mass"] == 0.001
