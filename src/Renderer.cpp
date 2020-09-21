#include <fstream>
#include "Scene.h"
#include "Renderer.h"
#include <algorithm>
#include <iostream>
#include <thread>
#include <mutex>
#include "Film.h"
#include "FilmTile.h"
#include "Bounds.h"


const float EPSILON = 0.00001;

class TileTask
{
public:
	void Init(int tileNum)
	{
		m_TileNum = tileNum;
		m_CurTile = 0;
	}

	struct Task
	{
		Task(bool hasTask, int tileId = -1) : m_HasTask(hasTask), m_TileId(tileId) {}

		bool m_HasTask;
		int m_TileId;
	};

	const Task GetTask()
	{
		std::lock_guard lg(m);

		if (m_CurTile >= m_TileNum)
		{
			return Task(false);
		}

		auto rst = Task(true, m_CurTile);
		m_CurTile++;

		return rst;
	}

private:
	int m_TileNum;
	int m_CurTile;
	std::mutex m;
};

// The main render function. This where we iterate over all pixels in the image,
// generate primary rays and cast these rays into the scene. The content of the
// framebuffer is saved to a file.
void Renderer::Render(const Scene& scene)
{
	int w = scene.width;
	int h = scene.height;

	auto film = Film(scene, "64spp.jpg");

	std::vector<Vector3f> framebuffer(w * h);
	std::vector<std::vector<Vector3f>> tempbuffer(h, std::vector<Vector3f>(w));

    float scale = tan(deg2rad(scene.fov * 0.5));
    float imageAspectRatio = scene.width / (float)scene.height;
    Vector3f eye_pos(278, 273, -800);

    // change the spp value to change sample ammount
    int spp = 1024;
    std::cout << "SPP: " << spp << "\n";

	const int threadNum = std::thread::hardware_concurrency();

	const int tileSize = 64;
	const int rowTiles = std::ceil(double(w) / tileSize);
	const int colTiles = std::ceil(double(h) / tileSize);
	const int tileNum = rowTiles * colTiles;
	TileTask tileTask;
	tileTask.Init(tileNum);
	
	auto renderPartImg = [&](int id)
	{
		for (auto task = tileTask.GetTask(); task.m_HasTask; task = tileTask.GetTask()) 
		{
			int tileID = task.m_TileId;
			int tileRow = tileID / rowTiles;
			int tileCol = tileID - tileRow * rowTiles;
			int x0 = tileCol * tileSize;
			int x1 = std::min(tileCol * tileSize + tileSize, w);
			int y0 = tileRow * tileSize;
			int y1 = std::min(tileRow * tileSize + tileSize, h);

			ResetRandom(tileID + 1);

			auto filmTile = film.GenFilmTile(Bounds2(Vector2f(x0, y0), Vector2f(x1, y1)));

			for (const auto pos : filmTile->AllPos())
			{
				int i = pos.x;
				int j = pos.y;

				float x = (2 * (i + 0.5) / (float)w - 1) * imageAspectRatio * scale;
				float y = (1 - 2 * (j + 0.5) / (float)h) * scale;

				Vector3f dir = normalize(Vector3f(-x, y, 1));
				Vector3f radiance = {};
				for (int k = 0; k < spp; k++)
				{
					radiance += (scene.castRay(Ray(eye_pos, dir), 0) / spp);
				}

				filmTile->AddSample(pos, radiance);
			}

			film.MergeFilmTile(filmTile);

			if (id == 0)
				UpdateProgress(tileID / (float)tileNum);
		}
	};

	std::vector<std::thread> workers;
	for (int i = 0; i < threadNum; i++)
		workers.push_back(std::thread(renderPartImg, i));

	// wait workers
	for (auto & worker : workers)
		worker.join();

	UpdateProgress(1.f);
	
	film.WriteImage();
}
