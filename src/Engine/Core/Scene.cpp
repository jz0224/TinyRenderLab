#include "Core/Scene.h"

void Scene::buildBVH() {
    printf(" - Generating BVH...\n\n");
    this->bvh = new BVHAccel(objects, 1, BVHAccel::SplitMethod::NAIVE);
}

Intersection Scene::intersect(const Ray &ray) const
{
    return this->bvh->Intersect(ray);
}

void Scene::sampleLight(Intersection &pos, float &pdf) const
{
    float emit_area_sum = 0;
    for (uint32_t k = 0; k < objects.size(); ++k) {
        if (objects[k]->hasEmit()){
            emit_area_sum += objects[k]->getArea();
        }
    }
    float p = GetRandomFloat() * emit_area_sum;
    emit_area_sum = 0;
    for (uint32_t k = 0; k < objects.size(); ++k) {
        if (objects[k]->hasEmit()){
            emit_area_sum += objects[k]->getArea();
            if (p <= emit_area_sum){
                objects[k]->Sample(pos, pdf);
                break;
            }
        }
    }
}

bool Scene::trace(
        const Ray &ray,
        const std::vector<Object*> &objects,
        float &tNear, uint32_t &index, Object **hitObject)
{
    *hitObject = nullptr;
    for (uint32_t k = 0; k < objects.size(); ++k) {
        float tNearK = kInfinity;
        uint32_t indexK;
        Vector2f uvK;
        if (objects[k]->intersect(ray, tNearK, indexK) && tNearK < tNear) {
            *hitObject = objects[k];
            tNear = tNearK;
            index = indexK;
        }
    }


    return (*hitObject != nullptr);
}

// Implementation of Path Tracing
Vector3f Scene::castRay(const Ray &ray, int depth) const
{
    //Implement Path Tracing Algorithm here
	Intersection intersection = intersect(ray);
	Vector3f hitcolor = Vector3f(0);

	//deal with light source
	if (intersection.emit.Length() > 0)
		hitcolor = Vector3f(1);
	else if (intersection.happened)
	{
		Vector3f wo = Normalize(-ray.direction);
		Vector3f p = intersection.coords;
		Vector3f N = Normalize(intersection.normal);

		float pdf_light = 0.0f;
		Intersection inter;
		sampleLight(inter, pdf_light);
		Vector3f x = inter.coords;
		Vector3f ws = Normalize(x - p);
		Vector3f NN = Normalize(inter.normal);

		Vector3f L_dir = Vector3f(0);
		//direct light
		if ((intersect(Ray(p, ws)).coords - x).Length() < 0.01)
		{
			L_dir = inter.emit * intersection.m->eval(wo, ws, N) * Dot(ws, N) * Dot(-ws, NN) / (((x - p).Length() * (x - p).Length()) * (pdf_light + EPSILON));
		}

		Vector3f L_indir = Vector3f(0);
		float P_RR = GetRandomFloat();
		//indirect light
		if (P_RR < Scene::RussianRoulette)
		{
			Vector3f wi = intersection.m->sample(wo, N);
			L_indir = castRay(Ray(p, wi), depth+1) * intersection.m->eval(wi, wo, N) * Dot(wi, N) / (intersection.m->pdf(wi, wo, N) * Scene::RussianRoulette + EPSILON);
		}
		hitcolor = L_indir + L_dir;
	}
	return hitcolor;
}