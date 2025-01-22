
# terminal ai (tai)

## use cases

```
tai whats the cache-control setting of google.com

$ curl -I https://google.com | grep -i 'cache-control'
Do you want to execute this command? [y/N] 
```

```
tai what is the cache-control header?

The Cache-Control HTTP header field is used to specify directives for caching mechanisms in both requests and responses. It is one of the most important headers fo
r controlling caching behavior. For example, to check the Cache-Control header of a website, you can use the command: `curl -I https://example.com | grep 'Cache-Co
ntrol'`
```

```
tai how do i send a json post via curl

To send a JSON POST request via curl, you can use the following command:

curl -X POST -H "Content-Type: application/json" -d '{"key":"value"}' https://example.com/api/endpoint

-X POST: Specifies the HTTP method as POST
-H "Content-Type: application/json": Sets the Content-Type header to application/json
-d '{"key":"value"}': Provides the JSON data to be sent in the request body
https://example.com/api/endpoint: The URL endpoint to send the request to
```

Examples:


```
$ cat article.txt | tai summarize the document
```

```
$ tai what headers is google.com returning?
---
curl -I https://google.com
Run the command? [y/N]:
```

```
$ tai "what's changed since last commit?"
$ tai "suggest a commit message based on my changes"
$ tai "explain this merge conflict"
```

```
$ tai "why is my CPU usage so high right now?"
$ tai "help me understand this log file pattern"
$ tai "suggest optimization for my nginx config"
```

```
$ tai analyze < main.py
```

```
$ cat logs.json | tai "find unusual patterns"
$ tai "convert this CSV to markdown table"
```

```
$ tai "convert these timestamps to my timezone"
$ tai "make this JSON pretty"
```

```
$ tai explain "awk '{print $1}'"
$ tai "what's the difference between -v and -vv in this command?"
```

