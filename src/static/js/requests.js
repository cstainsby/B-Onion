
// requests 
const initializeOpenAI = async () => {
  const url = '/openai/init';
  const data = { };
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
    return result.openai_response;
  } catch (error) {
    console.error(error);
  }
  return "";
}

const sendPromptToOpenAI = async (promptText, editions) => {
  const url = '/openai/prompt';
  const data = { 
    prompt_contents: promptText,
    editions: editions
  };
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
  return "";
}

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
