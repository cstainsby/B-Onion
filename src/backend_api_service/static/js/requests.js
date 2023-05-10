
// requests 
const postEditionToReddit = async (subreddit, edition) => {
  console.log("subreddit", subreddit, "edition", edition);
  const url = '/reddit/post';
  const data = {
    subreddit_name: subreddit,
    title: edition.title,
    content: edition.content
  }
  const options = {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(data)
  };

  try {
    const res = await fetch(url, options);
    const result = await res.json();
    return result;
  } catch (error) {
    console.error(error);
  }
}
