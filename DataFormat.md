# MAVEN dataset

Each `.jsonl` file is a subset of MAVEN and each line in the files is a json string for a document. For the `train.jsonl` and `valid.jsonl` the json format is as below:

```JSON5
{
    "id": "6b2e8c050e30872e49c2f46edb4ac044", // an unique string for each document
    "title": "Selma to Montgomery marches", // the tiltle of the document
    "content": [ // the content of the document. A list, each item is a dict for a sentence
    		{
    			"sentence": "...", // a string, the plain text of the sentence
    			"tokens": ["...", "..."] // a list, tokens of the sentence
		}
    ],
    "events":[ // a list for annotated events, each item is a dict for an event
        	{
            		"id": "75343904ec49aefe12c5749edadb7802", // an unique string for the event
            		"type": "Arranging", // the event type
            		"type_id": 70, // the numerical id for the event type
            		"mention":[ // a list for the event mentions of the event, each item is a dict
            			{
              				"id": "2db165c25298aefb682cba50c9327e4f", // an unique string for the event mention
              				"trigger_word": "organized", // a string of the trigger word or phrase
              				"sent_id": 1, // the index of the corresponding sentence, strates with 0
              				"offset": [3, 4], // the offset of the trigger words in the tokens list
              			}
             	     	]
        	},
    ],
    "negative_triggers":[ // a list for negative instances, each item is a dict for a negative mention
        {
        	"id": "46348f4078ae8460df4916d03573b7de",
            	"trigger_word": "desire",
            	"sent_id": 1,
            	"offset": [10, 11],
        },
    ]
}
```

For the `test.jsonl`, the format is almost the same but we hide the golden labels:

```JSON5
{
    "id": "6b2e8c050e30872e49c2f46edb4ac044", // an unique string for each document
    "title": "Selma to Montgomery marches", // the tiltle of the document
    "content": [ // the content of the document. A list, each item is a dict for a sentence
    		{
    		 	"sentence": "...", // a string, the plain text of the sentence
    		 	"tokens": ["...", "..."] // a list, tokens of the sentence
		}
    ],
    "candidates":[ // a list for trigger candidiates, each item is a dict for a trigger or a negative instance, you need to classify the type for each candidate
    	{
        	"id": "46348f4078ae8460df4916d03573b7de",
            	"trigger_word": "desire",
            	"sent_id": 1,
            	"offset": [10, 11],
        }
    ]
}
```

You can submit the prediction results for the test set to [CodaLab](https://competitions.codalab.org/competitions/27320) to get the test results. You need to name your result file as `results.jsonl` and compress it into a `.zip` file for submission.

Each line in the `results.jsonl` should be a json string encoding the prediction results for one document. The json format is as below:

```JSON5
{
	"id": "6b2e8c050e30872e49c2f46edb4ac044", // id for the document
  	"predictions":[ // a list, prediction results for the provided candidates
		{
			"id": "46348f4078ae8460df4916d03573b7de", // id for the candidate
			"type_id": 10, // integer id for the predicted type, 0 for the negative instances
		},
  	]
}
```
