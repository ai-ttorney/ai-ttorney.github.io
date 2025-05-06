const API_BASE_URL = import.meta.env.VITE_API_URL || '/api';

export const fetchAllThreads = async (userId: string) => {
  const response = await fetch(`${API_BASE_URL}/session/get_all_threads?user_id=${userId}`);
  return response.json();
};

export const createThread = async (userId: string) => {
  const response = await fetch(`${API_BASE_URL}/session/create_thread`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user_id: userId }),
  });
  return response.json();
};

export const deleteThreadById = async (threadId: string, userId: string) => {
  const response = await fetch(`${API_BASE_URL}/session/delete_thread`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ thread_id: threadId, user_id: userId }),
  });
  return response;
};

export const getThreadHistory = async (threadId: string, userId: string) => {
  const response = await fetch(`${API_BASE_URL}/session/get_thread_history`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ thread_id: threadId, user_id: userId }),
  });
  return response.json();
};

export const sendMessageToBackend = async (
  prompt: string,
  threadId: string,
  userId: string,
  isNewThread: boolean
) => {
  const body: any = {
    prompt,
    thread_id: threadId,
    user_id: userId
  };

  if (isNewThread) {
    body.is_new_thread = true;
  }

  const response = await fetch(`${API_BASE_URL}/chat/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  return response.json();
};
