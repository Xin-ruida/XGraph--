/*
   Modifications Copyright (C) 2018-2019 Keval Vora (keval@cs.sfu.ca)
   Copyright (c) 2014-2015 Xiaowei Zhu, Tsinghua University

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

#ifndef TYPE_H
#define TYPE_H

typedef int VertexId;
typedef long EdgeId;
typedef float Weight;

struct Empty { };

template <typename E>
struct Edge {
	VertexId source;
	VertexId target;
	E weight;
};

struct MergeStatus {
	int id;
	long begin_offset;
	long end_offset;
};

#endif
