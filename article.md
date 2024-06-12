Not Every Field is Just Text, Numbers or Vectors

![](https://lh7-us.googleusercontent.com/docsz/AD_4nXfiAhPMlvNUGxuXk8ocffQnzaPxzqxpazOhADXQ6m6LVQeGh2HwIIqg_2FQeHy5P3H9dVHHeFa8YTGsa2xnbrbmAU0eV7m1nS6zC-xGAr_a-rbA0tGyJjw6dcy0rEZBoPx0XciH_hmg09baGSqngr1T8HE?key=-9D8BcuPJ0m8A8eAGy9r0Q)

As I have been exploring all the features of Milvus, I found two interesting field types.  One for JSON and one for Arrays that let you store more complex data to augment your vectors.   In the roadmap, inverted indexes are coming for Arrays and JSON so these are types that you should start embracing and using in places where it makes sense.

JSON has been a widely used data format for a number of years, it has been used as a native storage format, used for processing in front ends, used as a data interchange format especially for RESTful endpoints and finally it is being embraced as a format for use in AI Agent flows.

See:   <https://developer.nvidia.com/blog/build-an-llm-powered-data-agent-for-data-analysis/>

One of the nice things about Milvus is that if we aren’t fully sure what our schema will be we can use a dynamic schema.   

When you use Dynamic Fields you are actually using a JSON Field.   The dynamic field in a collection is a reserved JSON field named $meta.

<https://milvus.io/docs/enable-dynamic-field.md>

I found a really good use for the JSON data field type when I started looking at Motor Vehicle Collision data for New York City which is available in a convenient frequently updated REST endpoint returning JSON.

Let’s start getting data!

**API INGEST**

For example, if we want to ingest information about the latest street cameras in New York City, that is a REST API with JSON data.  We did that recently and you can read about that one.

<https://medium.com/@tspann/unstructured-street-data-in-new-york-8d3cde0a1e5b>

So we see that it is pretty common to be working with JSON data, so fortunately we can handle that in Milvus and not need to put our data in multiple spots.

For our example today we will use Motor Vehicle Collisions in New York City.

<https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Crashes/h9gi-nx95/about_data>

You can take a look at the data dictionary here:

<https://data.cityofnewyork.us/api/views/h9gi-nx95/files/bd7ab0b2-d48c-48c4-a0a5-590d31a3e120?download=true&filename=MVCollisionsDataDictionary_20190813_ERD.xlsx>

With a REST endpoint that displays 1,000 records at a time.

[https://data.cityofnewyork.us/resource/h9gi-nx95.json?$limit=1000](https://data.cityofnewyork.us/resource/h9gi-nx95.json)

We can up the limit on this and also do paging.

![](https://lh7-us.googleusercontent.com/docsz/AD_4nXcdRUokgQsRoj-PICAaIMse9vRhYTt02aFO9ugDqsFmKX3EAY-iVFC6smHLU-jja2XNI6I-S_mQoW_Jsrj5sw6WqfLD-Vxx7TO9ACEnaYKAIJkLNBouWZ0RljOUCIs3E5tPuyZek2rBiNFd6mM5_Vp2Te8?key=-9D8BcuPJ0m8A8eAGy9r0Q)

**City of New York Open Data Documentation**

<https://dev.socrata.com/foundry/data.cityofnewyork.us/h9gi-nx95>

**NYC Open Data (Need to Sign Up)**

<https://data.cityofnewyork.us/profile/edit/developer_settings>

**How to Use Socrata Data**

<https://dev.socrata.com/>

**NYC Vision Zero View**

<https://vzv.nyc/>

**Example Usage**

<https://dev.socrata.com/blog/2019/10/07/time-series-analysis-with-jupyter-notebooks-and-socrata>

**Querying Giant Data Sets Via API**

<https://support.socrata.com/hc/en-us/articles/202949268-How-to-query-more-than-1000-rows-of-a-dataset>

<https://dev.socrata.com/docs/queries/limit.html>

<https://dev.socrata.com/docs/queries/offset.html>

<https://dev.socrata.com/docs/paging.html>

<https://dev.socrata.com/consumers/getting-started.html>

Let’s examine the REST endpoint first to make sure we are getting data back.

df = pd.read\_json('[https://data.cityofnewyork.us/resource/h9gi-nx95.json?$order=crash\_date+DESC&$limit=50](https://data.cityofnewyork.us/resource/h9gi-nx95.json?$order=crash_date+DESC&$limit=5000)')

**Example JSON Record**

{"crash\_date":"2021-09-11T00:00:00.000","crash\_time":"2:39","on\_street\_name":"WHITESTONE EXPRESSWAY","off\_street\_name":"20 AVENUE","number\_of\_persons\_injured":"2","number\_of\_persons\_killed":"0","number\_of\_pedestrians\_injured":"0","number\_of\_pedestrians\_killed":"0","number\_of\_cyclist\_injured":"0","number\_of\_cyclist\_killed":"0","number\_of\_motorist\_injured":"2","number\_of\_motorist\_killed":"0","contributing\_factor\_vehicle\_1":"Aggressive Driving/Road Rage","contributing\_factor\_vehicle\_2":"Unspecified","collision\_id":"4455765","vehicle\_type\_code1":"Sedan","vehicle\_type\_code2":"Sedan"}

For a crash free experience attend my meetups in Manhattan.

<https://www.meetup.com/unstructured-data-meetup-new-york/>

<https://lu.ma/calendar/manage/cal-VNT79trvj0jS8S7>

****![](https://lh7-us.googleusercontent.com/docsz/AD_4nXdlj_I9UbxoStGxAhAl1hbMMMH3NrEcbzcqBf5OB9KyoziK0sK47MDdZtqIwycDwuXTdkpCQosY00WZqJZ2urJhA16vgDjH4eLsUXU1II0fhJhF4x8tZD71bvRHSKqqIGS8VR9lDfFViwpGxOnCW6tI4u_I?key=-9D8BcuPJ0m8A8eAGy9r0Q)****

**ARRAY FIELDS**

Often data won’t be a simple string or number and may require an array of values.   Fortunately Milvus supports this.   

Adding a field to a schema that is an array is straightforward.

**field\_name**="ArrayFieldName"

You will need a field\_name like always, set this to whatever string makes sense for this field and its context.

**datatype**=DataType.ARRAY

For arrays, the data type must be **DataType.ARRAY**, not that this is surprising.

**element\_type**=DataType.INT64

For all elements they must match one data type, you can set this to any primitive type like Varchar, Int8, Int16, Int36, Int64, Bool, Float or Double.

**max\_capacity**=5

This is the maximum number of elements that your array can contain.   You can have less than this capacity or equal.   You cannot have more than this so this appropriately.   If your number of elements in your array varies greatly, you may have a lot of spare data here.   Choice arrays as a type carefully for your use case.

Let’s look at our notebook.

![](https://lh7-us.googleusercontent.com/docsz/AD_4nXdhB8TQk4uRTGyzGxik_eh6iYJ5C-NT9Aytu0F613Qixr0sWRyPdmmOd4Lj0fcTpLzAEFf7_pg20PC5ukgApJ-Xg5zJsXe7AeTevQMLHUwiwa7ypc6TCxW7IwJWzGuS9pFLCiAWUW11n41u50eC87cGVbE?key=-9D8BcuPJ0m8A8eAGy9r0Q)

**SOURCE CODE**

[**https://github.com/tspannhw/AIM-MotorVehicleCollisions/tree/main**](https://github.com/tspannhw/AIM-MotorVehicleCollisions/tree/main)

**NOTEBOOK**

[**https://github.com/tspannhw/AIM-MotorVehicleCollisions/blob/main/nycjson.ipynb**](https://github.com/tspannhw/AIM-MotorVehicleCollisions/blob/main/nycjson.ipynb)

**AGENT to LLM**

Now as we can see **JSON** is a pretty awesome and very useful type of field data type for Milvus but there is another field data type that is also very useful and they are **Arrays**.   An important differentiation between them is that Arrays must be all elements of the same type and have a fixed maximum capacity.   You can add less for future inserts.

<https://milvus.io/docs/array_data_type.md>

**ROADMAP**

<https://milvus.io/docs/roadmap.md>

In the upcoming Milvus 2.5, we will get to try out our new inverted indexes for Arrays and JSON, so I will update when that happens.   I will also go through the amazing list of new features and updates and give them a test run.

For a deeper diver into JSON and Array Data Types take a look at the resources below.

![](https://lh7-us.googleusercontent.com/docsz/AD_4nXdi2EkQuw9dPx2xl6ykoHQ_ZBAm1LuFZRDe_gscV2-TdXhaZ00mGTmI9BTtyzZe5Hpiujtj0HSmSJOrt0OhUXKmuOl6BgGPpE29rBp-sn2vSk3NwZrN6ThUxtCwvh1cthDdzEJOqV9j-seExYwIa959Ocou?key=-9D8BcuPJ0m8A8eAGy9r0Q)

**RESOURCES**

<https://zilliz.com/blog/json-metadata-filtering-in-milvus>

<https://milvus.io/docs/use-json-fields.md>

<https://milvus.io/docs/array_data_type.md>

<https://milvus.io/docs/schema.md>

<https://zilliz.com/blog/using-your-vector-database-as-JSON-or-relational-datastore> 

<https://medium.com/@zilliz_learn/vectorizing-json-data-with-milvus-for-similarity-search-1f546173162c>

<https://github.com/milvus-io/bootcamp/blob/master/bootcamp/Retrieval/imdb_metadata_json.ipynb>

<https://medium.com/@zilliz_learn/how-to-pick-a-vector-index-in-your-milvus-instance-a-visual-guide-2b6d9aa052c6>

<https://medium.com/@zilliz_learn/introduction-to-unstructured-data-68e4b3354d73>

<https://www.twitch.tv/vectordatabase>

<https://developer.nvidia.com/blog/building-your-first-llm-agent-application/>?

<https://github.com/NVIDIA/GenerativeAIExamples>

<https://milvus.io/api-reference/pymilvus/v2.4.x/ORM/Collection/insert.md>

<https://medium.com/@zilliz_learn/vectorizing-json-data-with-milvus-for-similarity-search-1f546173162c>

<https://milvus.io/docs/prepare-source-data.md>

<https://milvus.io/api-reference/pymilvus/v2.4.x/DataImport/LocalBulkWriter/LocalBulkWriter.md>

<https://milvus.io/docs/embed-with-sentence-transform.md>

**NOTES**

Make sure when you name your fields you keep them simple with only alphanumeric characters and underscores.     This one has gotten me before and you don’t want to keep changing your schema.   Make sure your ids match the type you use in your schema.   If your ID is a String and not an int64 you will get an error.

**TIP**

If you want to use Milvus Lite it does not currently work on ARM or Windows.

__![](https://lh7-us.googleusercontent.com/docsz/AD_4nXeS8gCWh2NF35ySk-QvfTJv6IQZSLS8Uf-d7eLSP7ChPB0xwOlJ-xjXiKNtXtjtYr10xOJ325SQIez7WWWmWR243lwXQjG_bwbcokNsTy2Bg6YWLRJt5n_3f9We2DVwCdkQGBEZKGBLlMC4G3q_lwItdfQS?key=-9D8BcuPJ0m8A8eAGy9r0Q)__

____
