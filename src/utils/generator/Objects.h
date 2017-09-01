/*
 * Copyright (c) 2014-2017 Christoph Niethammer <christoph.niethammer@gmail.com>
 *
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER
 */

#ifndef OBJECTS_H
#define OBJECTS_H

#include "utils/xmlfileUnits.h"

class Object {
public:
	Object() {}
	virtual void readXML(XMLfileUnits& xmlconfig) {}
	virtual ~Object() {}
	/** Determines if the given point is inside the object */
	virtual bool isInside(double r[3]) = 0;
	/** Determines if the given point is inside the object excluding it's border  */
	virtual bool isInsideNoBorder(double r[3]) = 0;
	/** Get lower corner of a bounding box around the object */
	virtual void getBboxMin(double rmin[3]) = 0;
	/** Get upper corner of a bounding box around the object */
	virtual void getBboxMax(double rmax[3]) = 0;
	/** Get name of object */
	virtual std::string getName() = 0;

	/* Forward getName to getPluginName required by pluginFactory template */
	virtual std::string getPluginName() final { return getName(); };
};

/** Class implementing a cuboid */
class Cuboid : public Object {
public:
	Cuboid();
	Cuboid(double lower[3], double upper[3]);
	void readXML(XMLfileUnits& xmlconfig);
	std::string getName() { return std::string("Cuboid"); }
	static Object* createInstance() { return new Cuboid(); }

	/** Determines if the given point is inside the object */
	bool isInside(double r[3]);
	/** Determines if the given point is inside the object excluding it's border  */
	bool isInsideNoBorder(double r[3]);
	/** Get lower corner of a bounding box around the object */
	void getBboxMin(double rmin[3]);
	/** Get upper corner of a bounding box around the object */
	void getBboxMax(double rmax[3]);
private:
	double _lowerCorner[3];
	double _upperCorner[3];
};

/** Class implementing a sphere */
class Sphere : public Object {
public:
	Sphere();
	Sphere(double center[3], double r);
	void readXML(XMLfileUnits& xmlconfig);
	std::string getName() { return std::string("Sphere"); }
	static Object* createInstance() { return new Sphere(); }

	bool isInside(double r[3]);

	bool isInsideNoBorder(double r[3]);

	void getBboxMin(double rmin[3]);
	void getBboxMax(double rmax[3]);

private:
	double _center[3];
	double _radius;
	double _radiusSquare;
};

/** Class implementing a cyliner */
class Cylinder : public Object {
public:
	Cylinder();
	/** Constructor
	 * @param[in]  centerBase   Center of the circle of the lower base of the cylinder.
	 * @param[in]  radius       Raius of the cylinder (x-y-axis)
	 * @param[in]  height       Height of the cylinder (z-axis)
	 */
	Cylinder(double centerBase[3], double radius, double height);
	void readXML(XMLfileUnits& xmlconfig);
	std::string getName() { return std::string("Cylinder"); }
	static Object* createInstance() { return new Cylinder(); }

	bool isInside(double r[3]);

	bool isInsideNoBorder(double r[3]);

	void getBboxMin(double rmin[3]);
	void getBboxMax(double rmax[3]);

private:
	double _radius;
	double _height;
	double _centerBase[3];
	double _radiusSquare;
};

/** Abstract class to unify two objects */
class ObjectUnification : public Object {
public:
	ObjectUnification();
	/** Constructor
	 * @param[in]  obj1  First object.
	 * @param[in]  obj2  Second object.
	 */
	ObjectUnification(Object *obj1, Object *obj2) : _ob1(obj1), _ob2(obj2) {}
	~ObjectUnification(){
		delete _ob1;
		delete _ob2;
	}

	void readXML(XMLfileUnits& xmlconfig);
	std::string getName() { return std::string("ObjectUnification"); }
	static Object* createInstance() { return new ObjectUnification(); }

	bool isInside(double r[3]) {
		return _ob1->isInside(r) || _ob2->isInside(r);
	}

	bool isInsideNoBorder(double r[3]) {
		/* either inside one of the objects or common border (intersection) */
		return _ob1->isInsideNoBorder(r) || _ob2->isInsideNoBorder(r) || (_ob1->isInside(r) && _ob2->isInside(r));
	}

	void getBboxMin(double rmin[3]) {
		double rmin1[3], rmin2[3];
		_ob1->getBboxMin(rmin1);
		_ob2->getBboxMin(rmin2);
		for(int d = 0; d < 3; d++) {
			rmin[d] = (rmin1[d] < rmin2[d]) ? rmin1[d] : rmin2[d] ;
		}
	}
	void getBboxMax(double rmax[3]) {
		double rmax1[3], rmax2[3];
		_ob1->getBboxMax(rmax1);
		_ob2->getBboxMax(rmax2);
		for(int d = 0; d < 3; d++) {
			rmax[d] = (rmax1[d] > rmax2[d]) ? rmax1[d] : rmax2[d] ;
		}
	}

private:
	Object* _ob1;
	Object* _ob2;
};

/** Abstract class to subtract one object from another */
class ObjectSubtractor : public Object {
public:
	ObjectSubtractor();
	/** Constructor
	 * @param[in]  original_ob  The original object.
	 * @param[in]  subtract_ob  The object which shall be subtract from the original object.
	 */
	ObjectSubtractor(Object *original_ob, Object *subtract_ob) : _ob1(original_ob), _ob2(subtract_ob) {}
	~ObjectSubtractor(){
		delete _ob1;
		delete _ob2;
	}

	void readXML(XMLfileUnits& xmlconfig);
	std::string getName() { return std::string("ObjectSubtractor"); }
	static Object* createInstance() { return new ObjectSubtractor(); }

	bool isInside(double r[3]) {
		return _ob1->isInside(r) && (!_ob2->isInsideNoBorder(r));
	}

	bool isInsideNoBorder(double r[3]) {
		return _ob1->isInsideNoBorder(r) && (!_ob2->isInside(r));
	}

	void getBboxMin(double rmin[3]) {
		_ob1->getBboxMin(rmin);
	}
	void getBboxMax(double rmax[3]) {
		_ob1->getBboxMax(rmax);
	}

private:
	Object* _ob1;
	Object* _ob2;
};

/** Abstract class for the intersection of two objects */
class ObjectIntersection : public Object {
public:
	ObjectIntersection();
	/** Constructor
	 * @param[in]  obj1  First object.
	 * @param[in]  obj2  Second object.
	 */
	ObjectIntersection(Object *obj1, Object *obj2) : _ob1(obj1), _ob2(obj2) {}
	~ObjectIntersection(){
		delete _ob1;
		delete _ob2;
	}

	void readXML(XMLfileUnits& xmlconfig);
	std::string getName() { return std::string("ObjectIntersection"); }
	static Object* createInstance() { return new ObjectIntersection(); }

	bool isInside(double r[3]) {
		return _ob1->isInside(r) && _ob2->isInside(r);
	}

	bool isInsideNoBorder(double r[3]) {
		return _ob1->isInsideNoBorder(r) && _ob2->isInsideNoBorder(r);
	}

	void getBboxMin(double rmin[3]) {
		double rmin1[3], rmin2[3];
		_ob1->getBboxMin(rmin1);
		_ob2->getBboxMin(rmin2);
		for(int d = 0; d < 3; d++) {
			rmin[d] = (rmin1[d] < rmin2[d]) ? rmin2[d] : rmin1[d] ;
		}
	}
	void getBboxMax(double rmax[3]) {
		double rmax1[3], rmax2[3];
		_ob1->getBboxMax(rmax1);
		_ob2->getBboxMax(rmax2);
		for(int d = 0; d < 3; d++) {
			rmax[d] = (rmax1[d] > rmax2[d]) ? rmax2[d] : rmax1[d] ;
		}
	}

private:
	Object* _ob1;
	Object* _ob2;
};

#endif /* OBJECTS_H */
